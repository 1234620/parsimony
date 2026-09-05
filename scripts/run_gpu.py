"""
GPU runner for the flagship 50M study (Kaggle T4 / Colab).

Stages (run independently so a 12h session limit is never a problem):
  --stage data      build + cache the training corpus and tokenizers
  --stage sweep     compute-matched vocabulary sweep at reduced scale
  --stage flagship  train the full ~50M model at the selected vocabulary

Every stage checkpoints and resumes.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from param_budget import solve_shape
from model import Parsimony, ParsimonyConfig
from data import SyntheticCorpus, TokenizedDataset
from tokenizer_train import train_tokenizer, load_tokenizer, corpus_stats, EOS_ID
from train import TrainConfig, train
from evaluate import bits_per_byte, reasoning_accuracy

DATA = ROOT / "data"
RESULTS = ROOT / "results"
CKPT = ROOT / "runs"
for p in (DATA, RESULTS, CKPT):
    p.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- data ----
def build_corpus(n_synth: int, target_docs: int) -> tuple[list[str], list[str]]:
    """Mixture: TinyStories (fluency) + FineWeb-Edu (knowledge) + our synthetic
    worked-reasoning traces. Each external source degrades to a warning rather
    than killing the run, so a flaky mirror never costs a GPU session."""
    texts: list[str] = []

    def try_hf(name, config, split, field, take):
        try:
            from datasets import load_dataset
            ds = load_dataset(name, config, split=split, streaming=True)
            got = []
            for i, ex in enumerate(ds):
                if i >= take:
                    break
                t = ex.get(field)
                if t and len(t) > 40:
                    got.append(t.strip())
            print(f"  [ok] {name}: {len(got):,} docs")
            return got
        except Exception as e:
            print(f"  [warn] {name} unavailable ({type(e).__name__}: {e}); continuing")
            return []

    print("building corpus...")
    texts += try_hf("roneneldan/TinyStories", None, "train", "text", int(target_docs * 0.55))
    texts += try_hf("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", int(target_docs * 0.25))

    gen = SyntheticCorpus(seed=7)
    synth = [gen.sample(mix=(0.15, 0.5, 0.35)) for _ in range(n_synth)]
    texts += synth
    print(f"  [ok] synthetic worked-reasoning traces: {len(synth):,} docs")

    if len(texts) < 1000:  # total external failure -> synthetic only
        print("  [warn] external sources unavailable; falling back to synthetic-only corpus")
        texts += [gen.sample() for _ in range(target_docs)]

    rng = np.random.default_rng(0)
    rng.shuffle(texts)
    holdout = texts[:600]
    return texts[600:], holdout


def reasoning_set(n: int = 200) -> list[dict]:
    g = SyntheticCorpus(seed=4242)
    items = []
    while len(items) < n:
        s = g.arithmetic() if len(items) % 2 == 0 else g.logic()
        if "Answer:" not in s:
            continue
        head, ans = s.rsplit("Answer:", 1)
        items.append({"prompt": head + "Answer:", "answer": ans.strip().rstrip(".")})
    return items


def tokenize_corpus(tok, texts) -> np.ndarray:
    flat: list[int] = []
    for t in texts:
        flat.extend(tok.encode(t).ids)
        flat.append(EOS_ID)
    return np.array(flat, dtype=np.int32)


# ------------------------------------------------------------- stages ----
def stage_data(args):
    texts, holdout = build_corpus(args.synth_docs, args.docs)
    (DATA / "train.txt").write_text("\n".join(t.replace("\n", "\\n") for t in texts), encoding="utf-8")
    (DATA / "holdout.txt").write_text("\n".join(t.replace("\n", "\\n") for t in holdout), encoding="utf-8")
    json.dump(reasoning_set(), (DATA / "reasoning.json").open("w"))
    print(f"train docs={len(texts):,}  holdout={len(holdout):,}")
    for v in args.vocabs:
        t0 = time.time()
        tok = train_tokenizer(texts[: args.tok_docs], v, DATA / f"tok_{v}.json")
        st = corpus_stats(tok, holdout[:400])
        print(f"  tokenizer {v}: vocab={st['actual_vocab']} bytes/token={st['bytes_per_token']:.3f} "
              f"({time.time()-t0:.0f}s)")


def _load_texts(p: Path) -> list[str]:
    return [l.replace("\\n", "\n") for l in p.read_text(encoding="utf-8").split("\n") if l]


def stage_sweep(args):
    train_texts = _load_texts(DATA / "train.txt")
    holdout = _load_texts(DATA / "holdout.txt")
    reason = json.load((DATA / "reasoning.json").open())
    out, results = RESULTS / "sweep_gpu", []

    for v in args.vocabs:
        tok = load_tokenizer(DATA / f"tok_{v}.json")
        st = corpus_stats(tok, holdout[:400])
        tokens = tokenize_corpus(tok, train_texts[: args.sweep_docs])
        ds = TokenizedDataset(tokens, args.block)

        shape = solve_shape(st["actual_vocab"], budget=args.sweep_budget, d_model=args.sweep_dmodel)
        cfg = ParsimonyConfig.from_shape(shape, max_seq_len=args.block)
        model = Parsimony(cfg)
        n_params, n_emb = model.num_parameters(), model.tok_emb.weight.numel()
        fpt = 6 * (n_params - n_emb) + 12 * cfg.n_layers * cfg.d_model * args.block
        steps = int(args.sweep_flops / (args.batch * args.block * fpt))
        print(f"\n[sweep v={v}] {shape.summary()}  steps={steps}")

        tcfg = TrainConfig(steps=steps, batch_size=args.batch, block_size=args.block,
                           lr=args.lr, eval_every=10**9, log_every=max(100, steps // 8),
                           ckpt_every=max(500, steps // 2), amp=True)
        info = train(model, ds, tcfg, device=DEVICE, out_dir=out / f"v{v}", resume=True)
        bpb = bits_per_byte(model, tok, holdout[:400], device=DEVICE, max_len=args.block, eos_id=EOS_ID)
        acc = reasoning_accuracy(model, tok, reason[:120], device=DEVICE, eos_id=EOS_ID)
        results.append({"vocab_actual": st["actual_vocab"], "n_layers": cfg.n_layers,
                        "params": n_params, "embed_fraction": n_emb / n_params,
                        "bytes_per_token": st["bytes_per_token"], "steps": steps,
                        "flops": info["total_flops"], "wall_s": info["wall_clock_s"],
                        **bpb, "reasoning_acc": acc["accuracy"]})
        print(f"  -> BPB={bpb['bits_per_byte']:.4f} reasoning={acc['accuracy']:.3f}")
        (out / "results.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(results, indent=2))

    best = min(results, key=lambda r: r["bits_per_byte"])
    print(f"\nbest vocabulary by bits/byte: {best['vocab_actual']} ({best['n_layers']} layers)")
    (RESULTS / "selected_vocab.json").write_text(json.dumps(best, indent=2))


def stage_flagship(args):
    sel = args.vocab or json.load((RESULTS / "selected_vocab.json").open())["vocab_actual"]
    tok_file = DATA / f"tok_{sel}.json"
    if not tok_file.exists():  # nearest trained tokenizer
        tok_file = min(DATA.glob("tok_*.json"),
                       key=lambda p: abs(int(p.stem.split("_")[1]) - sel))
    tok = load_tokenizer(tok_file)
    train_texts = _load_texts(DATA / "train.txt")
    holdout = _load_texts(DATA / "holdout.txt")
    reason = json.load((DATA / "reasoning.json").open())

    cache = DATA / f"tokens_{sel}.npy"
    if cache.exists():
        tokens = np.load(cache)
    else:
        tokens = tokenize_corpus(tok, train_texts)
        np.save(cache, tokens)
    print(f"training tokens: {len(tokens):,}")
    ds = TokenizedDataset(tokens, args.block)

    shape = solve_shape(tok.get_vocab_size(), budget=50_000_000, d_model=args.dmodel)
    cfg = ParsimonyConfig.from_shape(shape, max_seq_len=args.block)
    model = Parsimony(cfg)
    print(f"FLAGSHIP {shape.summary()}")
    print(f"verified parameter count: {model.num_parameters():,} (cap 50,000,000)")
    assert model.num_parameters() <= 50_000_000, "over budget"

    tcfg = TrainConfig(steps=args.steps, batch_size=args.batch, block_size=args.block,
                       lr=args.lr, eval_every=args.steps // 8, log_every=100,
                       ckpt_every=500, amp=True, grad_accum=args.grad_accum)

    def ev(m):
        r = bits_per_byte(m, tok, holdout[:200], device=DEVICE, max_len=args.block, eos_id=EOS_ID)
        return {"bpb": round(r["bits_per_byte"], 4), "ppl": round(r["token_ppl"], 2)}

    info = train(model, ds, tcfg, device=DEVICE, out_dir=CKPT / "flagship", resume=True, eval_fn=ev)
    bpb = bits_per_byte(model, tok, holdout, device=DEVICE, max_len=args.block, eos_id=EOS_ID)
    acc = reasoning_accuracy(model, tok, reason, device=DEVICE, eos_id=EOS_ID)
    summary = {"vocab": tok.get_vocab_size(), "params": model.num_parameters(),
               "d_model": cfg.d_model, "n_layers": cfg.n_layers,
               "train_tokens": info["total_tokens"], "total_flops": info["total_flops"],
               "wall_clock_s": info["wall_clock_s"],
               "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
               **bpb, "reasoning_acc": acc["accuracy"]}
    (RESULTS / "flagship.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "sweep", "flagship"])
    ap.add_argument("--vocabs", type=int, nargs="+", default=[2048, 8192, 16384, 32768, 50257])
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--docs", type=int, default=400_000)
    ap.add_argument("--synth-docs", type=int, default=120_000)
    ap.add_argument("--tok-docs", type=int, default=120_000)
    ap.add_argument("--sweep-docs", type=int, default=120_000)
    ap.add_argument("--sweep-budget", type=int, default=12_000_000)
    ap.add_argument("--sweep-dmodel", type=int, default=256)
    ap.add_argument("--sweep-flops", type=float, default=2.0e16)
    ap.add_argument("--dmodel", type=int, default=512)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    args = ap.parse_args()

    print(f"device={DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    {"data": stage_data, "sweep": stage_sweep, "flagship": stage_flagship}[args.stage](args)
