"""
Training loop with checkpoint/resume and explicit FLOP accounting.

Resume matters: free-tier GPU sessions get interrupted, and "training
efficiency" is a scored criterion, so we account for compute rather than
estimating it after the fact.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 16
    block_size: int = 256
    lr: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.02
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 250
    log_every: int = 50
    ckpt_every: int = 500
    seed: int = 1234
    amp: bool = True
    grad_accum: int = 1


def lr_at(step: int, cfg: TrainConfig) -> float:
    warmup = max(1, int(cfg.steps * cfg.warmup_ratio))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, cfg.steps - warmup)
    prog = min(1.0, max(0.0, prog))
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cos)


def build_optimizer(model, cfg: TrainConfig):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8,
    )


def model_flops_per_token(n_params_non_embed: int, n_layers: int, d_model: int, block: int) -> float:
    """6ND forward+backward, plus the attention term that 6ND omits."""
    attn = 12 * n_layers * d_model * block
    return 6 * n_params_non_embed + attn


def train(model, train_ds, cfg: TrainConfig, device="cpu", out_dir: Path = Path("runs/dev"),
          eval_fn=None, resume: bool = True, log_name: str = "log.jsonl"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model.to(device)
    opt = build_optimizer(model, cfg)
    use_amp = cfg.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_step = 0
    ckpt_path = out_dir / "ckpt.pt"
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        print(f"[resume] from step {start_step}")

    n_embed = model.tok_emb.weight.numel()
    n_non_embed = model.num_parameters() - n_embed
    fpt = model_flops_per_token(n_non_embed, model.cfg.n_layers, model.cfg.d_model, cfg.block_size)
    tokens_per_step = cfg.batch_size * cfg.block_size * cfg.grad_accum

    logf = (out_dir / log_name).open("a")
    t0 = time.time()
    hist = []

    for step in range(start_step, cfg.steps):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(cfg.grad_accum):
            xb, yb = train_ds.batch(cfg.batch_size, rng)
            xb = torch.from_numpy(xb).to(device, non_blocking=True)
            yb = torch.from_numpy(yb).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(xb, targets=yb)
                loss = loss / cfg.grad_accum
            scaler.scale(loss).backward() if use_amp else loss.backward()
            total_loss += loss.item()

        if use_amp:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if use_amp:
            scaler.step(opt); scaler.update()
        else:
            opt.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            el = time.time() - t0
            done = (step - start_step + 1)
            rec = {"step": step, "loss": total_loss, "lr": lr,
                   "tokens": (step + 1) * tokens_per_step,
                   "flops": (step + 1) * tokens_per_step * fpt,
                   "elapsed_s": el,
                   "tok_per_s": done * tokens_per_step / max(el, 1e-9)}
            hist.append(rec); logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(f"  step {step:>5} loss {total_loss:6.4f} lr {lr:.2e} "
                  f"{rec['tok_per_s']:>8.0f} tok/s")

        if eval_fn and (step + 1) % cfg.eval_every == 0:
            m = eval_fn(model)
            rec = {"step": step, "eval": m}
            hist.append(rec); logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(f"  step {step:>5} EVAL {m}")

        if (step + 1) % cfg.ckpt_every == 0 or step == cfg.steps - 1:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "cfg": asdict(cfg),
                        "model_cfg": asdict(model.cfg)}, ckpt_path)

    logf.close()
    total_tokens = cfg.steps * tokens_per_step
    return {"history": hist, "total_tokens": total_tokens,
            "total_flops": total_tokens * fpt,
            "wall_clock_s": time.time() - t0,
            "flops_per_token": fpt, "n_non_embed_params": n_non_embed}
