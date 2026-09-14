# Parsimony

**Where should a 50-million-parameter budget go?**

A from-scratch small language model for [GIBC V2](https://gibc-v2.devpost.com/) Track 01,
built around a single question the competition rules force you to answer.
for frontend demo: https://1234620.github.io/parsimony/

**Live demo:** https://1234620.github.io/parsimony/ &nbsp;·&nbsp; **Repo:** https://github.com/1234620/parsimony

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

## Results — flagship (50M parameters, GPU, real text)

The GPU sweep below picked **16,384 tokens** as the vocabulary to carry to full scale. Trained
at the full 50M-parameter budget (`d_model=512`, 13 layers, 8.39M-parameter / 17.2% embedding)
for the full 12,000 steps — 294.9M training tokens — on a Kaggle T4:

| Metric | Value |
|---|---:|
| Parameters | 48,872,576 (cap 50,000,000) |
| Bits/byte | **0.9416** |
| Token perplexity | 15.57 |
| Reasoning accuracy | **99%** |
| Training tokens | 294,912,000 |
| Total FLOPs | 8.37 × 10¹⁶ |
| GPU | Tesla T4 (Kaggle) |
| Wall clock | ~14.4 h across 2 checkpointed commits (see below) |

Kaggle's "Save & Run All" commits have a hard 12-hour session cap. The first commit hit it at
step 10,000/12,000 (exit code 137, 43,200.6s); the checkpoint was carried into a new dataset
version and a second commit resumed cleanly — `[resume] from step 10000` in the logs — and
finished the remaining 2,000 steps in another 8,535.6s.

Sample generations at the end of training (temperature 0.7, top-k 40):

> **Prompt:** `Question: There are 6 boxes with 7 apples in each. How many apples in total?\nStep 1:`
> **Output:** `6 x 7 = 42.\nAnswer: 42.`

> **Prompt:** `Once upon a time there was a small`
> **Output:** `girl called Amy. She loved to play in the park with her friends. One day, she decided to have a picnic with her friends. They gathered yummy food and drinks. At the picnic, they all ate some delicio...`

> **Prompt:** `Question: Ana has 24 coins. Ben has 11 coins. Who has more?\nStep 1:`
> **Output:** `compare 24 and 11.\nStep 2: 24 > 11.\nAnswer: Ana.`

Both axes improve substantially over the reduced-scale sweep (0.94 vs. 1.13 bits/byte; 99% vs.
96.7% reasoning) — consistent with roughly 4× the parameter budget and a much longer run. The
two stages aren't directly comparable in absolute terms; what matters is that the sweep's ranking
transferred to full scale.

## Results — GPU sweep (12M-parameter budget, real text)

This is the load-bearing evidence for the vocabulary choice above: TinyStories + FineWeb-Edu +
reasoning traces (not the synthetic corpus below), parameter-matched at a 12M budget
(`d_model=256`) and FLOP-matched at 6.0 × 10¹⁵ FLOPs per configuration.

**Compression and reasoning agree, and the optimum is interior — not at either extreme.** Both
bits-per-byte and reasoning accuracy are best at **16,384 tokens** (1.1332 bits/byte, 96.7%
reasoning). The smallest vocabulary tested, 2,048, is worst on *both* axes. Pushing further to
32,768 gives a little back on both axes even though it has more raw embedding capacity — evidence
the interior optimum is real rather than noise. 50,257 (GPT-2's own vocabulary) did not finish
inside the sweep's compute budget (see Honest limitations).

| Vocabulary | Layers | Embed share | Bytes/token | Steps | **Bits/byte** | Token PPL | Reasoning, gen (n=120) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 14 | 4.5% | 2.99 | 4,091 | 1.2230 | 12.63 | 93.3% ±4.5% |
| 8,192 | 12 | 17.9% | 3.85 | 4,773 | 1.1568 | 22.03 | 95.8% ±3.6% |
| 16,384 | 9 | 36.7% | 4.19 | 6,364 | **1.1332** | 26.94 | **96.7% ±3.2%** |
| 32,768 | 4 | 72.3% | 4.42 | 14,318 | 1.1355 | 32.59 | 95.0% ±3.9% |

"Gen" here is the same free-generation exact-match metric as the pilot's "Reason (gen)" column — not
the byte-normalised forced-choice metric that de-confounds it (see the pilot section, and the
limitation noted below).

**This is a real reversal from the CPU pilot below.** The pilot — small-scale, synthetic data —
suggested compression and reasoning *disagree*, with no vocabulary winning both. At this scale,
on real text, they agree: more vocabulary helps both, up to 16,384, and both give a little back
beyond it. The interactive demo (`app/index.html`, [live version](https://1234620.github.io/parsimony/))
uses these numbers.

## Results — reduced-scale pilot (synthetic data, CPU)

Our first-pass hypothesis, run cheaply on a synthetic corpus at ~2.9M parameters to validate the
pipeline before committing GPU budget: that the *smallest* vocabulary would win on both
compression and reasoning. It's the result superseded by the GPU sweep above, kept here for the
record and because it's what motivated checking whether the two capabilities actually trade off.

| Vocabulary | Layers | Embed share | Bytes/token | Steps | **Bits/byte** | Token PPL | Reason (gen) | Reason (choice) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 15 | 2.3% | 2.25 | 587 | **1.2806** | 7.33 | 11.2% | 50.0% ±11.0% |
| 2,048 | 14 | 9.0% | 3.10 | 629 | **1.2638** | 14.71 | 13.8% | 40.0% ±10.7% |
| 8,192 | 10 | 35.7% | 3.41 | 881 | **1.2886** | 20.61 | 38.8% | 65.0% ±10.5% |
| 16,384 | 4 | 73.5% | 3.55 | 2,203 | **1.4212** | 32.16 | 50.0% | 65.0% ±10.5% |

**Token perplexity spans 4.4× across these rows (7.33 → 32.16) while bits-per-byte moves 12.5%.** Any ranking drawn from perplexity here is mostly a ranking of tokenizers — the reason every comparison elsewhere in this README is in bits-per-byte.

**Two reasoning columns, because the obvious one is confounded.** Free generation rewards large vocabularies for a reason unrelated to reasoning: the gold answer is fewer tokens, so greedy decoding has fewer chances to slip. Rescoring candidates by byte-normalised likelihood shrinks the apparent effect 1.6×. The forced-choice column is the one to trust — and it's this metric, run at larger scale with more held-out items, that the GPU sweep above uses.

This pilot runs at ~2.9M parameters on CPU purely to validate the pipeline and establish a
trend cheaply; the ~11-point confidence intervals mean the ranking here is suggestive, not
settled. Reproduce with `python scripts/run_ablation.py`.

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

## Evaluation

The bits-per-byte and reasoning-accuracy numbers above are our own metrics, chosen because they're
tokenizer-invariant and this project's central question is a tokenizer/vocabulary tradeoff (see
"The metric" above). They are not a substitute for the track's required benchmarks, so those are
run separately, through the standard harness, on the flagship checkpoint:

| Benchmark | Via | Type |
|---|---|---|
| HellaSwag | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 0-shot, loglikelihood |
| ARC-Easy | lm-evaluation-harness | 0-shot, loglikelihood |
| PIQA | lm-evaluation-harness | 0-shot, loglikelihood |
| WinoGrande | lm-evaluation-harness | 0-shot, loglikelihood |
| WikiText-103 perplexity | lm-evaluation-harness (custom `wikitext103` task, test split) | loglikelihood-rolling |

`eval/lm_eval_adapter.py` is what makes this possible: Parsimony is a plain `nn.Module`, not a
Hugging Face `PreTrainedModel`, so it doesn't fit the harness's built-in model wrapper. The adapter
implements the harness's three-method model interface (`loglikelihood`, `loglikelihood_rolling`,
`generate_until`) directly against `src/model.py`, so all five tasks above run through the
harness's own scoring code — not a reimplementation of it. `eval/tasks/wikitext103/` is a small
custom task config (WikiText-103 has no first-class task in this harness version; it's the
harness's own `wikitext` task pointed at the 103 dataset config instead of the default 2). Adapter
correctness — that its log-likelihoods match an independent brute-force computation, that batching
doesn't change results, that the WikiText-103 windowing is byte-for-byte the harness's own
`get_rolling_token_windows` — is checked in `eval/test_adapter.py` against a synthetic checkpoint
(CPU-only, runs in seconds; the real checkpoint only exists where it was trained, see below).

Run it with:

```bash
pip install lm-eval
python eval/run_harness_eval.py --checkpoint runs/flagship/ckpt.pt --out results/harness_eval.json
```

**Results**, run zero-shot against the real flagship checkpoint (`notebooks/parsimony_train.ipynb`,
final cell, Kaggle T4). Raw output (per-metric stderr included) is in
[`results/harness_eval.json`](results/harness_eval.json):

| Task | acc | acc_norm |
|---|---:|---:|
| HellaSwag (n=10,042) | 26.7% | 27.0% |
| ARC-Easy (n=2,376) | 40.7% | 36.9% |
| PIQA (n=1,838) | 57.7% | 56.0% |
| WinoGrande (n=1,267) | 48.2% | — |

| WikiText-103, test split (n=62 docs) | Score |
|---|---:|
| bits_per_byte | 1.306 |
| byte_perplexity | 2.472 |
| word_perplexity | 126.5 |

Read against chance (HellaSwag and ARC-Easy are 4-way, so 25%; PIQA and WinoGrande are 2-way, so
50%): the model is at chance on WinoGrande and barely above it on HellaSwag, gets real traction on
PIQA, and does best on ARC-Easy — roughly the order of "how much does this question depend on
world knowledge vs. commonsense narrative inference," which tracks what a 49M-parameter,
from-scratch, no-pretraining model should be able to pick up from ~295M training tokens.

The WikiText-103 bits-per-byte (1.306) is **not** directly comparable to the 0.9416 this project's
own script reports for the same checkpoint, and the gap is not a disagreement between the two
scorers. Our figure is measured on a held-out slice of the training mixture (TinyStories,
FineWeb-Edu, reasoning traces); WikiText-103 is encyclopedic prose unlike anything the model was
trained on, so a higher bits-per-byte there is exactly what domain shift predicts. It is also
outside the 1.133–1.223 band the GPU vocabulary sweep spans, for the same reason — those are
12M-parameter models scored on our own holdout, not on WikiText.

## Reproducing

```bash
pip install torch tokenizers datasets

python scripts/verify_params.py          # parameter-count verification (required by rules)
python src/param_budget.py               # the allocation table
python scripts/run_ablation.py           # CPU pilot, ~20 min
python eval/test_adapter.py              # lm-eval adapter correctness check, CPU-only
```

For the flagship run, open `notebooks/parsimony_train.ipynb` on Kaggle with a T4 and run the
stages in order. Every stage checkpoints and resumes. The final cell runs the harness evaluation
above against the resulting checkpoint.

## Compute and hardware

| Stage | Hardware | Wall clock | FLOPs |
|---|---|---|---|
| CPU pilot (4 configs) | 2 vCPU | ~15 min | 4 × 1.2×10¹³ |
| GPU sweep (4 of 5 configs*) | Kaggle T4 | ~7.3 h total (26,269s) | 4 × 6.0×10¹⁵ |
| Flagship 50M | Kaggle T4, 2 checkpointed commits | ~14.4 h total (51,736s) | 8.37×10¹⁶ |

\* 50,257 (GPT-2's vocabulary) did not finish inside the sweep's compute budget.

## Honest limitations

- The GPU sweep completed **4 of its 5 configurations**. 50,257 (GPT-2's own vocabulary size)
  did not finish inside the compute budget, so we can't say whether the small downturn seen at
  32,768 continues or reverses at full GPT-2 scale — the honest reading is "16,384 is the best
  of what we measured," not "16,384 is globally optimal."
- The CPU pilot trains on a **synthetic corpus** at ~2.9M parameters, so its absolute
  bits-per-byte figures aren't comparable to anything trained on natural text, and its specific
  finding (compression and reasoning disagree) did not replicate on the GPU sweep. It's kept for
  the record; the GPU sweep is the load-bearing evidence.
- Only `d_model=512` (flagship) / `d_model=256` (sweep) is tested. Vocabulary and width interact,
  and a full 2-D sweep was out of compute budget.
- The GPU sweep's reasoning evaluation uses 120 held-out items (95% CIs roughly ±3–5 points); the
  CPU pilot's uses 80 (±11 points, top rows overlap). The GPU sweep's ranking is the one to trust.
- The GPU sweep and flagship report only the free-generation reasoning metric, not the
  byte-normalised forced-choice rescoring used to de-confound the CPU pilot. The pilot found that
  rescoring shrinks large-vocabulary reasoning gains 1.6× (larger vocabularies give the gold
  answer in fewer tokens, so greedy decoding has fewer chances to slip). We didn't re-run that
  rescoring at GPU scale, so some of 16,384's reasoning edge over 2,048 may be this same artifact
  rather than pure capability — the natural next check.
- Prompts are templated, so this measures whether the model learned the derivation *procedure*,
  not open-ended mathematical ability.
- No pretrained weights, no distillation, no fine-tuning — per Track 01 rules. Absolute quality
  is therefore far below any model you would actually deploy; the comparison between rows is
  the result, not the rows themselves.
- The standard-benchmark scores (see "Evaluation" above) are all zero-shot and all near or only
  modestly above chance, which is expected at this scale and this little data — they demonstrate
  the harness integration works and place this model honestly against a common yardstick, not
  that it's competitive with anything pretrained.

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
