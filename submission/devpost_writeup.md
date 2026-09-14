# Parsimony — Devpost submission copy

Paste each block into the matching field on the GIBC V2 submission form. Flagship numbers
below are filled in from the completed Kaggle run (`results/flagship.json`).

---

## Project name
Parsimony

## Tagline
Half a 50M-parameter model is a dictionary. We measured what that costs — and what it buys.

## Track
Track 01 — Foundational LLM (trained from scratch, ≤50M parameters)

---

## Inspiration

The Track 01 rules contain one clause that changes the whole problem: the 50M parameter cap
**includes token embeddings and the output head**.

Once we did the arithmetic, the competition stopped looking like "train a small model" and
started looking like a resource-allocation problem. At `d_model=512`, GPT-2's 50,257-token
vocabulary costs 25.7M parameters — **54% of the entire budget** — before a single transformer
layer exists. That leaves room for 7 layers. An 8,192-token vocabulary costs 4.2M and leaves
room for 14.

Same parameter count. Double the depth. We could not find anyone who had measured which of
those is actually the better model, so we did.

## What it does

Parsimony is a decoder-only language model trained from scratch, plus the experimental apparatus
that answers the allocation question:

- a **budget solver** that, for any vocabulary size, derives the deepest model fitting under 50M
  parameters and reports exactly where every parameter went;
- a **compute-matched training harness** that gives every configuration the same FLOPs, not the
  same steps;
- a **tokenizer-invariant evaluation** in bits-per-byte, plus length-normalized forced-choice
  reasoning scoring;
- an **interactive demo** where dragging a vocabulary slider reallocates the budget live and
  transformer layers visibly appear and disappear.

## What we found

Our hypothesis, from a cheap synthetic-data pilot, was that a small vocabulary wins because
depth is where computation happens — the pilot found the two capabilities we measured
*disagreeing*, with no vocabulary winning both. **The GPU sweep on real text overturned that:
at the scale that matters, compression and reasoning agree, and both peak at the same,
fairly large vocabulary.**

Across a parameter- and FLOP-matched sweep of {2,048, 8,192, 16,384, 32,768}-token
vocabularies on TinyStories, FineWeb-Edu, and our reasoning traces, both bits-per-byte
(1.1332) and reasoning accuracy (96.7%) are best at **16,384 tokens** — not the smallest
vocabulary we tested (worst on both) and not the largest we completed (32,768 gives a little
back on both). GPT-2's own 50,257-token vocabulary didn't finish training inside our compute
budget, so we can only say 16,384 is the best of what we measured, not that it's globally
optimal.

We then trained that winning configuration at the full 50M-parameter budget for the complete
12,000-step run: **0.9416 bits/byte, 99% reasoning accuracy, 48,872,576 parameters** — both
well ahead of any reduced-scale sweep config, as expected from roughly 4× the parameters and a
much longer run. Kaggle's 12-hour session cap killed the first training commit at step
10,000/12,000; the checkpoint/resume path we built for exactly this reason picked it up
cleanly and finished the remaining 2,000 steps in a second commit.

One confound we're upfront about: both the sweep and the flagship report free-generation
reasoning accuracy, not the byte-normalized forced-choice rescoring we used in the pilot to
catch a tokenization artifact there (larger vocabularies give the gold answer in fewer tokens,
so greedy decoding has fewer chances to slip — rescoring shrank that pilot's apparent effect
1.6×). We didn't re-run that rescoring at GPU scale, so part of 16,384's edge over 2,048 above
may be this same artifact rather than pure capability. We'd rather name that gap than paper
over it.

## How we built it

**Architecture** — deliberately conventional-modern, so the variable under study is the
allocation and not a pile of confounding novelties: RMSNorm pre-norm, RoPE, SwiGLU with
`d_ff = 8/3·d`, grouped-query attention with QK-norm, tied input/output embeddings, and residual
projections initialized at `1/sqrt(2L)`.

Initialization was verified against theory rather than assumed: loss at init **9.065** against
`ln(V) = 9.011`, logit standard deviation **0.456** against the predicted `sqrt(d)·0.02 = 0.453`.

**Data** — TinyStories for fluency (33M-parameter models write coherently on it), FineWeb-Edu
for clean world knowledge, and procedurally generated worked-reasoning traces that show the
*derivation* rather than only the answer.

**Method** — every configuration is parameter-matched and FLOP-matched. Evaluation is in
bits-per-byte, normalized by raw bytes of held-out text, because per-token perplexity cannot be
compared across tokenizers.

## Challenges we ran into

**BPE saturated silently.** Our first synthetic corpus had too few distinct word types, so every
requested vocabulary above ~1k collapsed to the same 753-merge tokenizer. Three "different"
configurations were secretly the identical model, and the results table looked plausible. We
caught it only because two rows were byte-identical. Fixed by rebuilding the generator over a
56,990-word lexicon.

**Uniform sampling would have rigged the study.** With every word equally likely, a large
vocabulary has no frequent word types to capture, so it looks useless by construction. Real
language is Zipfian. We weight sampling by 1/rank — without that fix the headline result would
have been an artifact of our own data generator.

**Perplexity is a trap here.** Token perplexity varies several-fold across our sweep while
bits-per-byte moves only slightly. Anyone comparing tokenizers on perplexity is measuring the
tokenizer, not the model.

## What we learned

That our own hypothesis, and the small pilot that seemed to support it, were both wrong in an
interesting way — not because the reasoning was bad, but because a synthetic corpus at 1/17th
scale doesn't reliably preview what happens on real text at full scale. The interesting question
in a constrained-budget regime isn't "which architecture is best," it's "does my cheap proxy
experiment actually predict my expensive one" — and here, it partially didn't.

Also: the discipline that mattered most was not any modelling trick. It was refusing to compare
numbers that were not comparable — across tokenizers, across step counts, across answer lengths —
and being willing to say plainly when a confound (like free-generation length bias) hadn't been
ruled out at full scale, rather than letting a clean headline number stand unqualified.

## What's next

A two-dimensional sweep over vocabulary *and* width — the two interact and we could only afford
one axis. Factorized embeddings (`V×r` then `r×d`) are implemented but unswept, and offer a
middle path: large vocabulary coverage at a fraction of the parameter cost.

## Built with

Python, PyTorch, HuggingFace `tokenizers`, NumPy, TinyStories, FineWeb-Edu, Kaggle T4 GPUs.
Demo: vanilla JavaScript and inline SVG, no framework.

## Try it

- Interactive allocator demo: https://1234620.github.io/parsimony/
- Repository: https://github.com/1234620/parsimony
- Parameter verification: `python scripts/verify_params.py`
