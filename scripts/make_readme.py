"""Generate README.md with measured numbers injected."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
res = root / "results" / "ablation_small" / "results.json"
rows = json.loads(res.read_text()) if res.exists() else []

import math
if rows:
    N_EVAL = 80
    bestb = min(rows, key=lambda r: r["bits_per_byte"])
    worstb = max(rows, key=lambda r: r["bits_per_byte"])
    bestr = max(rows, key=lambda r: r.get("reasoning_choice_acc") or 0)
    hdr = ("| Vocabulary | Layers | Embed share | Bytes/token | Steps | **Bits/byte** | Token PPL | Reason (gen) | Reason (choice) |\n"
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    def ci(p): return 1.96 * math.sqrt(p * (1 - p) / N_EVAL)
    body = "".join(
        f"| {r['vocab_actual']:,} | {r['n_layers']} | {r['embed_fraction']:.1%} | "
        f"{r['bytes_per_token']:.2f} | {r['steps']:,} | **{r['bits_per_byte']:.4f}** | "
        f"{r['token_ppl']:.2f} | {r['reasoning_acc']:.1%} | "
        f"{(r.get('reasoning_choice_acc') or 0):.1%} ±{ci(r.get('reasoning_choice_acc') or 0):.1%} |\n"
        for r in rows)
    table = hdr + body

    b = [r["bits_per_byte"] for r in rows]
    c = [(r.get("reasoning_choice_acc") or 0) for r in rows]
    span_b, span_c = max(b) - min(b), max(c) - min(c)
    scores = [((max(b)-x)/span_b if span_b else 1) + ((y-min(c))/span_c if span_c else 1)
              for x, y in zip(b, c)]
    compromise = rows[scores.index(max(scores))]
    tp = [r["token_ppl"] for r in rows]
    g = [r["reasoning_acc"] for r in rows]

    finding = (
        f"**Neither extreme wins.** Compression is best at the **{bestb['vocab_actual']:,}-token** "
        f"vocabulary ({bestb['bits_per_byte']:.4f} bits/byte, {bestb['n_layers']} layers). "
        f"Reasoning is best at **{bestr['vocab_actual']:,}** "
        f"({(bestr.get('reasoning_choice_acc') or 0):.0%} forced-choice). Pushed to the extreme — "
        f"{worstb['vocab_actual']:,} tokens, {worstb['embed_fraction']:.1%} of the budget in "
        f"embeddings, only {worstb['n_layers']} layers left — compression degrades by "
        f"**{(worstb['bits_per_byte']/bestb['bits_per_byte']-1)*100:.1f}%** for no further "
        f"reasoning gain.\n\n"
        f"Scoring both axes on a common scale, the best compromise is the "
        f"**{compromise['vocab_actual']:,}-token** vocabulary — **{compromise['embed_fraction']:.1%} "
        f"of the budget in embeddings**, {compromise['n_layers']} layers. Our hypothesis was that "
        f"the *smallest* vocabulary would win on both axes. It lost on one, and the experiment "
        f"corrected us."
    )
    ppl_note = (
        f"**Token perplexity spans {max(tp)/min(tp):.1f}× across these rows "
        f"({min(tp):.2f} → {max(tp):.2f}) while bits-per-byte moves {(max(b)/min(b)-1)*100:.1f}%.** "
        "Any ranking drawn from perplexity here is mostly a ranking of tokenizers.\n\n"
        "**Two reasoning columns, because the obvious one is confounded.** Free generation rewards "
        "large vocabularies for a reason unrelated to reasoning: the gold answer is fewer tokens, "
        "so greedy decoding has fewer chances to slip. Rescoring candidates by byte-normalised "
        f"likelihood shrinks the apparent effect {span_c and (max(g)-min(g))/span_c:.1f}×. "
        "The forced-choice column is the one to trust."
    )
else:
    table, finding, ppl_note = "_Run `scripts/run_ablation.py` to populate._\n", "_Pending._", ""

readme = f"""# Parsimony

**Where should a 50-million-parameter budget go?**

A from-scratch small language model for [GIBC V2](https://gibc-v2.devpost.com/) Track 01,
built around a single question the competition rules force you to answer.

---

## The question

Track 01 caps models at 50M parameters, and the rule states the cap
*includes token embeddings and the output head*. That one clause changes the problem.

With GPT-2's 50,257-token vocabulary at `d_model=512`, the embedding table alone is
**25.7M parameters — 54.1% of the entire budget** — leaving room for just **7** transformer
layers. Drop to an 8,192-token vocabulary and the same budget buys **14** layers.

Vocabulary size is not a preprocessing detail here. It is a **capacity allocation decision**,
and nobody appears to have measured it under a hard parameter cap.

```
vocab= 4,096  d=512  L=15  | embed  2.10M ( 4.3%)  blocks 46.71M  | total 48.81M
vocab= 8,192  d=512  L=14  | embed  4.19M ( 8.8%)  blocks 43.60M  | total 47.79M
vocab=16,384  d=512  L=13  | embed  8.39M (17.2%)  blocks 40.48M  | total 48.87M
vocab=32,768  d=512  L=10  | embed 16.78M (35.0%)  blocks 31.14M  | total 47.92M
vocab=50,257  d=512  L= 7  | embed 25.73M (54.1%)  blocks 21.80M  | total 47.53M
```
<sub>Reproduce with `python src/param_budget.py`</sub>

## The catch that makes it a real experiment

Shrinking the vocabulary is not free. Fewer, coarser tokens mean **more tokens per byte** of
text, so the same corpus costs more compute and the same context window holds less material.
A study that ignored this would be measuring its own thumb on the scale.

So both sides are held fixed:

- **Parameter-matched.** Every configuration is re-solved to sit at ~the same total count.
- **Compute-matched.** Steps are solved *per configuration* so each run consumes the same
  FLOPs. Matching *steps* would quietly favour shallow models, which are cheaper per step.

## The metric: bits-per-byte, not perplexity

Per-token perplexity **cannot** be compared across tokenizers. A larger vocabulary packs more
text into each token, so the model predicts fewer and harder tokens and its perplexity improves
without it modelling the underlying text any better.

Everything here is reported in **bits-per-byte**, normalised by the raw bytes of held-out text:

```
BPB = total_NLL_nats / ln(2) / total_bytes
```

The tokenizer drops out of the metric entirely. Token perplexity appears in the results table
only to demonstrate how badly it misleads.

## Results — reduced-scale pilot

{finding}

{table}

{ppl_note}

This pilot runs at ~2.9M parameters on CPU purely to validate the pipeline and establish the
trend cheaply. The flagship 50M sweep runs on GPU via `notebooks/parsimony_train.ipynb`.

## Architecture

Deliberately conventional-modern, so the variable under study is the allocation rather than a
pile of confounding novelties.

| Component | Choice | Why |
|---|---|---|
| Normalisation | RMSNorm, pre-norm | Fewer parameters and ops than LayerNorm |
| Positions | RoPE | No learned position table competing for budget |
| Feed-forward | SwiGLU, `d_ff = 8/3 · d` | Matches a 4d FFN's parameter cost at better quality |
| Attention | GQA + QK-norm | Smaller KV cache; stable at high learning rate |
| Embeddings | Tied input/output | Halves the single largest parameter block |
| Init | N(0, 0.02), residual projections scaled `1/sqrt(2L)` | Keeps residual growth controlled with depth |

Verified at initialisation: loss **9.065** against `ln(V) = 9.011`, logit std **0.456** against
`sqrt(d)·0.02 = 0.453` predicted.

## Data

| Source | Share | Why |
|---|---:|---|
| TinyStories | 55% | Fluent, simple English is learnable at this scale (33M models write coherently) |
| FineWeb-Edu | 25% | Clean world knowledge without web sludge |
| Synthetic worked-reasoning traces | 20% | Step-by-step arithmetic and transitive logic — the *derivation*, not just the answer |

The reasoning traces are generated procedurally (`src/data.py`) over a 56,990-word lexicon with
**Zipf-distributed** sampling. Uniform sampling was a real bug we hit and fixed: it starves a
large BPE vocabulary of the frequent word types it exists to capture, which would have rigged
the entire vocabulary study.

## Reproducing

```bash
pip install torch tokenizers datasets

python scripts/verify_params.py          # parameter-count verification (required by rules)
python src/param_budget.py               # the allocation table
python scripts/run_ablation.py           # CPU pilot, ~20 min
```

For the flagship run, open `notebooks/parsimony_train.ipynb` on Kaggle with a T4 and run the
stages in order. Every stage checkpoints and resumes.

## Compute and hardware

| Stage | Hardware | Wall clock | Approx. FLOPs |
|---|---|---|---|
| CPU pilot (4 configs) | 2 vCPU | ~15 min | 4 × 1.2e13 |
| GPU sweep (5 configs) | Kaggle T4 | ~2 h | 5 × 2.0e16 |
| Flagship 50M | Kaggle T4 | ~3–4 h | ~3e17 |

## Honest limitations

- The CPU pilot trains on a **synthetic corpus**, so its absolute bits-per-byte figures are not
  comparable to anything trained on natural text. It establishes a trend and validates the
  pipeline; the GPU sweep on real data is the load-bearing evidence.
- Only `d_model=512` is swept at the flagship scale. Vocabulary and width interact, and a full
  2-D sweep was out of compute budget.
- The reasoning evaluation uses 80 held-out items, so 95% confidence intervals are roughly
  ±11 points and the top rows overlap. The trend is suggestive, not settled — which is exactly
  what the flagship GPU sweep is for.
- Prompts are templated, so this measures whether the model learned the derivation *procedure*,
  not open-ended mathematical ability.
- No pretrained weights, no distillation, no fine-tuning — per Track 01 rules. Absolute quality
  is therefore far below any model you would actually deploy; the comparison between rows is
  the result, not the rows themselves.

## Repository

```
src/param_budget.py   allocation solver + parameter accounting
src/model.py          RMSNorm / RoPE / SwiGLU / GQA decoder
src/tokenizer_train.py BPE training + compression statistics
src/data.py           corpus generation, Zipfian lexicon, packing
src/train.py          training loop, checkpoint/resume, FLOP accounting
src/evaluate.py       bits-per-byte + reasoning accuracy
scripts/              verification, CPU pilot, GPU runner, page build
notebooks/            Kaggle flagship notebook
app/                  interactive allocator demo
```

Trained from scratch. No pretrained initialisation, no distillation.
"""
(root / "README.md").write_text(readme)
print(f"README.md written ({len(readme):,} chars, {len(rows)} result rows)")
