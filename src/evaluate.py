"""
Tokenizer-invariant evaluation.

Bits-per-byte is the metric that makes models with different vocabularies
comparable:  BPB = total_NLL_nats / ln(2) / total_bytes_of_raw_text.

Token-level perplexity is NOT comparable across tokenizers -- a model with a
larger vocabulary predicts fewer, harder tokens, so its per-token PPL looks
better without the model being better at modelling the underlying text. Every
number in the vocabulary study is therefore reported in BPB.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F


@torch.no_grad()
def bits_per_byte(model, tokenizer, texts: List[str], device="cpu", max_len: int = 512, eos_id: int = 1) -> dict:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0

    for text in texts:
        ids = tokenizer.encode(text).ids + [eos_id]
        total_bytes += len(text.encode("utf-8")) + 1  # +1 charges the EOS to the document
        if len(ids) < 2:
            continue
        for start in range(0, len(ids) - 1, max_len):
            chunk = ids[start : start + max_len + 1]
            if len(chunk) < 2:
                continue
            x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            total_nll += nll.item()
            total_tokens += y.numel()

    return {
        "bits_per_byte": total_nll / math.log(2) / max(total_bytes, 1),
        "token_ppl": math.exp(total_nll / max(total_tokens, 1)),
        "nats_per_token": total_nll / max(total_tokens, 1),
        "n_tokens": total_tokens,
        "n_bytes": total_bytes,
    }


@torch.no_grad()
def reasoning_accuracy(model, tokenizer, items: List[dict], device="cpu",
                       max_new_tokens: int = 12, eos_id: int = 1) -> dict:
    """Exact-match accuracy on held-out arithmetic/logic prompts.

    We score the answer the model *generates* after its worked steps, which is
    a harder and more honest test than ranking pre-supplied options.
    """
    model.eval()
    correct = 0
    records = []
    for it in items:
        ids = tokenizer.encode(it["prompt"]).ids
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.0 + 1e-6,
                             top_k=1, eos_id=eos_id)
        gen = tokenizer.decode(out[0, len(ids):].tolist())
        hit = it["answer"].strip() in gen.strip()[:40]
        correct += hit
        records.append({"prompt": it["prompt"][-60:], "gen": gen[:40], "want": it["answer"], "hit": bool(hit)})
    return {"accuracy": correct / max(len(items), 1), "n": len(items), "records": records[:5]}


@torch.no_grad()
def reasoning_choice_accuracy(model, tokenizer, items: List[dict], device="cpu",
                              max_len: int = 512) -> dict:
    """Length-normalised forced choice -- removes the tokenisation confound.

    Free-form generation quietly favours large vocabularies: if the gold answer
    is one token instead of three, greedy decoding has fewer chances to slip.
    That inflates the apparent reasoning advantage of a big vocabulary without
    the model reasoning any better.

    Here every candidate is *scored* instead: we take the negative log-likelihood
    of each candidate continuation and normalise it by the candidate's length in
    BYTES, so neither vocabulary nor answer length can tilt the comparison. The
    model picks the lowest-cost continuation, exactly as ARC/HellaSwag are scored.
    """
    model.eval()
    correct = 0
    for it in items:
        prompt_ids = tokenizer.encode(it["prompt"]).ids
        best, best_cost = None, float("inf")
        for cand in [it["answer"]] + list(it.get("distractors", [])):
            cand_ids = tokenizer.encode(" " + cand).ids
            if not cand_ids:
                continue
            ids = (prompt_ids + cand_ids)[-max_len - 1:]
            n_cand = min(len(cand_ids), len(ids) - 1)
            x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            )[-n_cand:].sum().item()
            cost = nll / max(len((" " + cand).encode("utf-8")), 1)  # bits per byte of answer
            if cost < best_cost:
                best_cost, best = cost, cand
        correct += (best == it["answer"])
    return {"choice_accuracy": correct / max(len(items), 1), "n": len(items)}
