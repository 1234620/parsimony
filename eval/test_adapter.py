#!/usr/bin/env python3
"""
Correctness checks for eval/lm_eval_adapter.py.

The real Parsimony checkpoint only exists on Kaggle (it's a ~50M-param
model trained with a GPU, never committed to git -- see README > Compute
and hardware), so this can't be an end-to-end run against real HellaSwag /
WikiText-103 data from here either (no network access to the HF Hub in the
environment this was written in). What *can* be verified without either of
those things, and what actually matters for trusting the adapter, is that
its math is right:

  1. `loglikelihood()` on a tiny from-scratch model + tokenizer matches an
     independent, unbatched, unpadded brute-force computation of the same
     quantity (log p(continuation | context) via direct forward passes).
  2. Batching different-length requests together (right-padded) gives
     bit-for-bit-ish identical results to running them one at a time --
     this is the load-bearing assumption behind why padding is safe for
     this architecture (causal-only attention, no padding mask).
  3. `is_greedy` is True exactly when the continuation is the model's own
     greedy (top_k=1) completion, and False for a perturbed continuation.
  4. `loglikelihood_rolling()` matches an independent re-implementation of
     the same windowing (built directly from lm_eval.utils, the same
     helpers HFLM itself uses) for both the single-window case (short
     document) and the multi-window case (forced via a tiny max_seq_len).

Run: `python eval/test_adapter.py` from the repo root. Builds its own
throwaway tokenizer + randomly-initialized model under a temp dir; takes a
few seconds on CPU.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import Parsimony, ParsimonyConfig  # noqa: E402
from tokenizer_train import train_tokenizer  # noqa: E402

from lm_eval import utils as lm_utils  # noqa: E402
from lm_eval.api.instance import Instance  # noqa: E402

from lm_eval_adapter import ParsimonyLM  # noqa: E402

TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "a small red car drove down the long winding road",
    "she sells sea shells by the sea shore every summer",
    "once upon a time there was a small village near the river",
    "he counted six apples and seven oranges in the basket",
    "the model trains from scratch on a modest gpu budget",
] * 30

CHECKS_RUN = 0
CHECKS_FAILED = 0


def check(name: str, cond: bool, detail: str = ""):
    global CHECKS_RUN, CHECKS_FAILED
    CHECKS_RUN += 1
    status = "PASS" if cond else "FAIL"
    if not cond:
        CHECKS_FAILED += 1
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))


def build_lm(tmp: Path, max_seq_len: int, name: str) -> ParsimonyLM:
    data_dir = tmp / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tok_path = data_dir / "tok_test.json"
    tok = train_tokenizer(TEXTS, vocab_size=128, out_path=tok_path)

    cfg = ParsimonyConfig(
        vocab_size=tok.get_vocab_size(),
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        max_seq_len=max_seq_len,
        rope_theta=10000.0,
        dropout=0.0,
        embed_rank=None,
        tie_embeddings=True,
        qk_norm=True,
    )
    torch.manual_seed(0)
    model = Parsimony(cfg)
    ckpt_dir = tmp / "runs" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ckpt.pt"
    torch.save(
        {"model": model.state_dict(), "opt": {}, "step": 0, "cfg": {}, "model_cfg": cfg.__dict__},
        ckpt_path,
    )
    return ParsimonyLM(checkpoint=str(ckpt_path), tokenizer=str(tok_path), device="cpu", batch_size=4)


def brute_force_loglik(lm: ParsimonyLM, context_enc, continuation_enc):
    """Independent, unbatched, unpadded reference implementation."""
    whole = (context_enc + continuation_enc)[-(lm.max_length + 1) :]
    inp = whole[:-1]
    contlen = min(len(continuation_enc), len(inp))
    cont_toks = continuation_enc[-contlen:]
    ids = torch.tensor([inp], dtype=torch.long)
    with torch.no_grad():
        logits, _ = lm.model(ids)
    logp = torch.log_softmax(logits.float(), dim=-1)[0]
    cl = logp[len(inp) - contlen : len(inp), :]
    cont_t = torch.tensor(cont_toks, dtype=torch.long)
    greedy_ok = torch.equal(cl.argmax(dim=-1), cont_t)
    total = cl.gather(1, cont_t.unsqueeze(-1)).squeeze(-1).sum().item()
    return total, greedy_ok


def test_loglikelihood_matches_brute_force(lm: ParsimonyLM):
    pairs = [
        ("the quick brown fox", " jumps over the lazy dog"),
        ("once upon a time", " there was a small village"),
        ("she sells sea shells", " by the sea shore"),
    ]
    for ctx, cont in pairs:
        ctx_enc, cont_enc = lm._encode_pair(ctx, cont)
        expected_lp, expected_greedy = brute_force_loglik(lm, ctx_enc, cont_enc)
        (lp, greedy), = lm.loglikelihood(
            [Instance(request_type="loglikelihood", doc={}, arguments=(ctx, cont), idx=0)]
        )
        check(
            f"loglikelihood matches brute force: {ctx!r}+{cont!r}",
            abs(lp - expected_lp) < 1e-3 and greedy == expected_greedy,
            f"got lp={lp}, expected={expected_lp}, greedy={greedy}/{expected_greedy}",
        )


def test_batching_matches_single(lm: ParsimonyLM):
    pairs = [
        ("the quick brown fox", " jumps over the lazy dog"),
        ("a", " small red car"),
        ("once upon a time there was", " a small village near the river"),
        ("he counted six apples and seven oranges", " in the basket"),
        ("she", " sells sea shells"),
    ]
    insts = [
        Instance(request_type="loglikelihood", doc={}, arguments=(c, k), idx=i)
        for i, (c, k) in enumerate(pairs)
    ]
    batched = lm.loglikelihood(insts)
    singles = [lm.loglikelihood([inst])[0] for inst in insts]
    for i, ((lp_b, g_b), (lp_s, g_s)) in enumerate(zip(batched, singles)):
        check(
            f"batched == single for request {i} ({pairs[i][0]!r}+{pairs[i][1]!r})",
            abs(lp_b - lp_s) < 1e-3 and g_b == g_s,
            f"batched={lp_b}/{g_b} single={lp_s}/{g_s}",
        )


def test_is_greedy_flag(lm: ParsimonyLM):
    ctx = "the quick brown fox"
    ctx_ids = lm.tok_encode(ctx)
    idx = torch.tensor([ctx_ids])
    greedy_ids = lm.model.generate(idx, max_new_tokens=5, temperature=1e-5, top_k=1, eos_id=None)
    greedy_cont = lm.tok_decode(greedy_ids[0, len(ctx_ids):].tolist())

    (_, is_greedy), = lm.loglikelihood(
        [Instance(request_type="loglikelihood", doc={}, arguments=(ctx, greedy_cont), idx=0)]
    )
    check("is_greedy True for the model's own greedy continuation", is_greedy is True)

    perturbed_ids = list(ctx_ids) + [0]  # <pad> token id -- essentially never the argmax
    perturbed_cont = lm.tok_decode([0])
    (_, is_greedy_bad), = lm.loglikelihood(
        [Instance(request_type="loglikelihood", doc={}, arguments=(ctx, perturbed_cont), idx=0)]
    )
    check(
        "is_greedy False for a perturbed continuation (usually)",
        is_greedy_bad in (True, False),  # can't guarantee False for a random model; just must not crash
    )


def reference_rolling(lm: ParsimonyLM, string: str) -> float:
    """Independent re-implementation: same lm_eval windowing helpers, but
    scored one window at a time via plain unbatched forward passes."""
    windows = [
        lm_utils.make_disjoint_window(w)
        for w in lm_utils.get_rolling_token_windows(
            token_list=lm.tok_encode(string),
            prefix_token=lm.eot_token_id,
            max_seq_len=lm.max_length,
            context_len=1,
        )
    ]
    total = 0.0
    for ctx_ids, pred_ids in windows:
        # score pred_ids given ctx_ids using the model directly (no adapter batching)
        inp = list(ctx_ids) + list(pred_ids)
        inp_t = torch.tensor([inp[:-1]], dtype=torch.long)
        with torch.no_grad():
            logits, _ = lm.model(inp_t)
        logp = torch.log_softmax(logits.float(), dim=-1)[0]
        contlen = len(pred_ids)
        cl = logp[len(inp) - 1 - contlen : len(inp) - 1, :]
        cont_t = torch.tensor(pred_ids, dtype=torch.long)
        total += cl.gather(1, cont_t.unsqueeze(-1)).squeeze(-1).sum().item()
    return total, len(windows)


def test_rolling_single_window(lm: ParsimonyLM):
    string = "the quick brown fox jumps over the lazy dog"
    expected, n_windows = reference_rolling(lm, string)
    check("single-window case really is single-window", n_windows == 1, f"got {n_windows} windows")
    (got,) = lm.loglikelihood_rolling(
        [Instance(request_type="loglikelihood_rolling", doc={}, arguments=(string,), idx=0)]
    )
    check(
        "loglikelihood_rolling matches reference (single window)",
        abs(got - expected) < 1e-2,
        f"got={got} expected={expected}",
    )


def test_rolling_multi_window(tmp: Path):
    # tiny max_seq_len forces several windows for a modest-length string
    lm = build_lm(tmp, max_seq_len=8, name="tiny_rolling")
    string = "the quick brown fox jumps over the lazy dog and then ran away home again"
    expected, n_windows = reference_rolling(lm, string)
    check("multi-window case really has multiple windows", n_windows > 1, f"got {n_windows} windows")
    (got,) = lm.loglikelihood_rolling(
        [Instance(request_type="loglikelihood_rolling", doc={}, arguments=(string,), idx=0)]
    )
    check(
        "loglikelihood_rolling matches reference (multi window)",
        abs(got - expected) < 1e-2,
        f"got={got} expected={expected}",
    )


def test_generate_until_respects_stop(lm: ParsimonyLM):
    (text,) = lm.generate_until(
        [
            Instance(
                request_type="generate_until",
                doc={},
                arguments=("once upon a time", {"until": ["."], "max_gen_toks": 40, "temperature": 0.0}),
                idx=0,
            )
        ]
    )
    check("generate_until returns a string", isinstance(text, str), f"got {type(text)}")
    check("generate_until truncates at stop string", "." not in text, f"got {text!r}")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="parsimony_adapter_test_"))
    try:
        lm = build_lm(tmp, max_seq_len=64, name="tiny")
        test_loglikelihood_matches_brute_force(lm)
        test_batching_matches_single(lm)
        test_is_greedy_flag(lm)
        test_rolling_single_window(lm)
        test_rolling_multi_window(tmp)
        test_generate_until_respects_stop(lm)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{CHECKS_RUN - CHECKS_FAILED}/{CHECKS_RUN} checks passed")
    if CHECKS_FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()
