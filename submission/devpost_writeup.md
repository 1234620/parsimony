# Parsimony — Devpost submission copy

Paste each block into the matching field on the GIBC V2 submission form.
Numbers marked `[FLAGSHIP]` come from `results/flagship.json` after your Kaggle run.

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

Our hypothesis was that a small vocabulary wins, because depth is where computation happens.
**The experiment refuted it, and the real answer is more interesting: the two capabilities we
measured disagree.**

Bits-per-byte is nearly flat across the entire sweep — compression barely cares how the budget
is split. Reasoning accuracy is not flat at all. Spending a large share of the budget on the
embedding table costs very little compression and changes reasoning substantially.

We also caught ourselves nearly reporting a confounded result. Free-form generation appeared to
favour large vocabularies dramatically — but with a bigger vocabulary the gold answer is fewer
tokens, so greedy decoding simply has fewer chances to slip. Rescoring every candidate with
byte-normalized likelihood (the way ARC and HellaSwag are scored) removed that advantage and
changed the ranking. Both numbers are reported side by side, because the gap between them is
itself a finding about how small-model reasoning gets measured.

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

That the interesting question in a constrained-budget regime is not "which architecture is best"
but "which capability are you buying." At 50M parameters those trade against each other, and the
answer depends on what the model is for.

Also: the discipline that mattered most was not any modelling trick. It was refusing to compare
numbers that were not comparable — across tokenizers, across step counts, across answer lengths.
Two of our three near-miss errors were measurement errors, not modelling errors.

## What's next

A two-dimensional sweep over vocabulary *and* width — the two interact and we could only afford
one axis. Factorized embeddings (`V×r` then `r×d`) are implemented but unswept, and offer a
middle path: large vocabulary coverage at a fraction of the parameter cost.

## Built with

Python, PyTorch, HuggingFace `tokenizers`, NumPy, TinyStories, FineWeb-Edu, Kaggle T4 GPUs.
Demo: vanilla JavaScript and inline SVG, no framework.

## Try it

- Interactive allocator demo: `[YOUR DEPLOYED URL]`
- Repository: `[YOUR GITHUB URL]`
- Parameter verification: `python scripts/verify_params.py`
