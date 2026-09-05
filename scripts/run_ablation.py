"""
Compute-matched vocabulary allocation study (reduced scale).

Design:
  - FIXED parameter budget: every config sits at ~the same total param count.
  - FIXED compute budget: steps are solved per-config so each run consumes the
    same FLOPs. Matching *steps* would unfairly favour shallow configs, which
    are cheaper per step -- so we match FLOPs instead.
  - Metric: bits-per-byte on held-out raw text (tokenizer-invariant), plus
    exact-match accuracy on held-out reasoning prompts.

This is the small-scale version of the flagship 50M sweep; it establishes the
trend cheaply so GPU time is spent only on the configuration it selects.
"""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from param_budget import solve_shape
from model import Parsimony, ParsimonyConfig
from data import SyntheticCorpus, TokenizedDataset
from tokenizer_train import train_tokenizer, corpus_stats, EOS_ID
from train import TrainConfig, train
from evaluate import bits_per_byte, reasoning_accuracy, reasoning_choice_accuracy

torch.set_num_threads(2)

BUDGET       = int(os.environ.get("BUDGET", 3_000_000))
D_MODEL      = int(os.environ.get("D_MODEL", 128))
FLOP_BUDGET  = float(os.environ.get("FLOP_BUDGET", 1.2e13))
VOCABS       = [int(v) for v in os.environ.get("VOCABS", "512,2048,8192,16384").split(",")]
BLOCK        = 128
BATCH        = 8
OUT          = Path("results/ablation_small")
OUT.mkdir(parents=True, exist_ok=True)

print("== building corpus ==", flush=True)
gen = SyntheticCorpus(seed=0)
train_texts = [gen.sample() for _ in range(30000)]
val_gen = SyntheticCorpus(seed=999)
val_texts = [val_gen.sample() for _ in range(300)]

# held-out reasoning prompts: cut the worked solution, ask for the answer
reason_gen = SyntheticCorpus(seed=555)
reason_items = []
while len(reason_items) < 80:
    s = reason_gen.arithmetic() if len(reason_items) % 2 == 0 else reason_gen.logic()
    if "Answer:" not in s:
        continue
    head, ans = s.rsplit("Answer:", 1)
    ans = ans.strip().rstrip(".")
    if ans.lstrip("-").isdigit():                       # numeric: near-miss distractors
        n = int(ans)
        dis = [str(n + d) for d in (1, -1, 10) if n + d != n and n + d >= 0][:3]
    else:                                               # name: other names from the prompt
        cands = [w.strip(".,?") for w in head.split()
                 if w[:1].isupper() and w.strip(".,?") not in (ans, "Question:", "Step", "Answer:", "Who")]
        dis = list(dict.fromkeys(cands))[:3]
    if len(dis) < 2:
        continue
    reason_items.append({"prompt": head + "Answer:", "answer": ans, "distractors": dis})

print(f"distinct word types={len(set(chr(32).join(train_texts).split())):,}")
print(f"train docs={len(train_texts)}  val docs={len(val_texts)}  reasoning items={len(reason_items)}", flush=True)

results = []
for vocab in VOCABS:
    t_start = time.time()
    print(f"\n===== vocab {vocab} =====", flush=True)
    tok_path = OUT / f"tok_{vocab}.json"
    tok = train_tokenizer(train_texts, vocab, tok_path)
    stats = corpus_stats(tok, val_texts[:100])
    print(f"  tokenizer: actual_vocab={stats['actual_vocab']} bytes/token={stats['bytes_per_token']:.3f}", flush=True)

    ids = [tok.encode(t).ids for t in train_texts]
    flat = []
    for seq in ids:
        flat.extend(seq); flat.append(EOS_ID)
    tokens = np.array(flat, dtype=np.int32)
    ds = TokenizedDataset(tokens, BLOCK)
    print(f"  train tokens={len(tokens):,}", flush=True)

    shape = solve_shape(stats["actual_vocab"], budget=BUDGET, d_model=D_MODEL)
    cfg = ParsimonyConfig.from_shape(shape, max_seq_len=BLOCK)
    model = Parsimony(cfg)
    n_params = model.num_parameters()
    n_embed = model.tok_emb.weight.numel()

    # compute-matched steps
    fpt = 6 * (n_params - n_embed) + 12 * cfg.n_layers * cfg.d_model * BLOCK
    steps = int(FLOP_BUDGET / (BATCH * BLOCK * fpt))
    print(f"  {shape.summary()}", flush=True)
    print(f"  params={n_params:,} flops/token={fpt:.3e} -> {steps} steps (compute-matched)", flush=True)

    tcfg = TrainConfig(steps=steps, batch_size=BATCH, block_size=BLOCK, lr=3e-3,
                       eval_every=10**9, log_every=max(50, steps // 6),
                       ckpt_every=10**9, amp=False)
    info = train(model, ds, tcfg, device="cpu", out_dir=OUT / f"run_v{vocab}", resume=False)

    bpb = bits_per_byte(model, tok, val_texts, device="cpu", max_len=BLOCK, eos_id=EOS_ID)
    acc = reasoning_accuracy(model, tok, reason_items, device="cpu", max_new_tokens=8, eos_id=EOS_ID)
    choice = reasoning_choice_accuracy(model, tok, reason_items, device="cpu", max_len=BLOCK)

    rec = {"vocab_requested": vocab, "vocab_actual": stats["actual_vocab"],
           "d_model": cfg.d_model, "n_layers": cfg.n_layers, "params": n_params,
           "embed_params": n_embed, "embed_fraction": n_embed / n_params,
           "bytes_per_token": stats["bytes_per_token"],
           "steps": steps, "train_tokens": info["total_tokens"],
           "flops": info["total_flops"], "wall_s": info["wall_clock_s"],
           "final_train_loss": info["history"][-1].get("loss"),
           **{k: v for k, v in bpb.items()},
           "reasoning_acc": acc["accuracy"], "reasoning_choice_acc": choice["choice_accuracy"],
           "reasoning_samples": acc["records"][:3]}
    results.append(rec)
    print(f"  -> BPB={bpb['bits_per_byte']:.4f}  tokPPL={bpb['token_ppl']:.2f}  "
          f"gen={acc['accuracy']:.3f} choice={choice['choice_accuracy']:.3f}  ({time.time()-t_start:.0f}s)", flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

print("\n==== SUMMARY (compute- and parameter-matched) ====", flush=True)
print(f"{'vocab':>7} {'L':>3} {'emb%':>6} {'B/tok':>7} {'steps':>6} {'BPB':>8} {'tokPPL':>9} {'gen':>6} {'choice':>7}", flush=True)
for r in results:
    print(f"{r['vocab_actual']:>7} {r['n_layers']:>3} {r['embed_fraction']:>5.1%} "
          f"{r['bytes_per_token']:>7.3f} {r['steps']:>6} {r['bits_per_byte']:>8.4f} "
          f"{r['token_ppl']:>9.2f} {r['reasoning_acc']:>6.3f} {r['reasoning_choice_acc']:>7.3f}", flush=True)
(OUT / "results.json").write_text(json.dumps(results, indent=2))
print("\nwrote", OUT / "results.json", flush=True)
