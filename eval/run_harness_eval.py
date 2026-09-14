#!/usr/bin/env python3
"""
Run the official Track 01 evaluation: HellaSwag, ARC-Easy, PIQA, WinoGrande
(via lm-evaluation-harness) plus WikiText-103 perplexity, against a trained
Parsimony checkpoint.

Usage (from the repo root, with a GPU and the checkpoint present -- this is
meant to be run on Kaggle as the last cell of notebooks/parsimony_train.ipynb
once MODE == "flagship" has produced runs/flagship/ckpt.pt):

    python eval/run_harness_eval.py \
        --checkpoint runs/flagship/ckpt.pt \
        --out results/harness_eval.json

All four required tasks are zero-shot (num_fewshot=0), matching how the
rubric's benchmarks are conventionally reported for small from-scratch
models. Pass --limit N while iterating to sanity-check on a handful of
examples per task before committing to a full (slow) run.

This script is the "evaluation script you used" the README's Evaluation
section points to; see eval/lm_eval_adapter.py for the model wrapper that
does the actual work.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs/flagship/ckpt.pt")
    ap.add_argument("--tokenizer", default=None, help="defaults to data/tok_*.json")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=float, default=None, help="cap docs per task; for quick smoke tests")
    ap.add_argument("--out", default="results/harness_eval.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(REPO_ROOT / "src"))

    import lm_eval
    from lm_eval.tasks import TaskManager

    from lm_eval_adapter import ParsimonyLM

    print(f"Loading Parsimony checkpoint from {args.checkpoint} ...")
    lm = ParsimonyLM(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        batch_size=args.batch_size,
        device=args.device,
    )
    n_params = lm.model.num_parameters()
    print(f"Loaded. {n_params:,} trainable params, max_seq_len={lm.max_length}, device={lm.device}")

    task_manager = TaskManager(include_path=str(Path(__file__).resolve().parent / "tasks"))

    print(f"Running: {args.tasks} (num_fewshot={args.num_fewshot}, limit={args.limit})")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        task_manager=task_manager,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_params": n_params,
        "checkpoint": args.checkpoint,
        "tasks": args.tasks,
        "num_fewshot": args.num_fewshot,
        "results": results["results"],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {out_path}")

    print("\n=== Summary (paste into README > Evaluation) ===")
    for task, metrics in results["results"].items():
        line = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
            if not k.endswith("_stderr,none") and k != "alias"
        )
        print(f"{task}: {line}")


if __name__ == "__main__":
    main()
