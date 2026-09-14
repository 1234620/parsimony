"""
lm-evaluation-harness adapter for Parsimony.

The hackathon rubric scores submissions on HellaSwag, ARC-Easy, PIQA and
WinoGrande "via lm-evaluation-harness", plus perplexity on a held-out slice
of WikiText-103, with the evaluation script included in the repo. This file
*is* that script's core: it makes the trained Parsimony checkpoint speak the
one interface the harness (github.com/EleutherAI/lm-evaluation-harness,
package `lm-eval`) needs from any model, so the four required tasks run
through the harness's own standard scoring code -- not a custom metric.

Why a custom adapter at all: Parsimony is a small nn.Module trained from
scratch (src/model.py), not a Hugging Face `PreTrainedModel`, so it doesn't
fit lm-eval's built-in `hf` model wrapper. `lm_eval.api.model.TemplateLM`
is the documented extension point for exactly this case -- subclass it,
implement three methods, and every existing task (including the four
required here) works unmodified:

  * `_loglikelihood_tokens` -- scores a batch of (context, continuation)
    token pairs. This is what HellaSwag / ARC-Easy / PIQA / WinoGrande run
    on: each is a multiple-choice task that ranks candidate continuations
    by log-likelihood.
  * `loglikelihood_rolling` -- full-document log-likelihood via a
    non-overlapping sliding window, token-exact with lm_eval's own
    `utils.get_rolling_token_windows` / `make_disjoint_window`. This is
    what the harness's perplexity tasks (including the WikiText-103 task
    registered in eval/tasks/wikitext103/) use internally, so implementing
    it correctly is the entire WikiText-103 requirement -- no separate
    perplexity script needed.
  * `generate_until` -- greedy-ish free-form generation with stop strings.
    Not exercised by the four required tasks (they're all loglikelihood
    tasks) but implemented for completeness / future tasks.

See eval/run_harness_eval.py for the driver that actually calls
`lm_eval.simple_evaluate(...)` with this model, and eval/test_adapter.py
for adapter-level correctness checks (run against a tiny synthetic
checkpoint, since the real one only exists on Kaggle -- see that file's
docstring for exactly what is and isn't verified).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import Parsimony, ParsimonyConfig  # noqa: E402
from tokenizer_train import EOS_ID, load_tokenizer  # noqa: E402

from lm_eval import utils as lm_utils  # noqa: E402
from lm_eval.api.model import TemplateLM  # noqa: E402
from lm_eval.api.registry import register_model  # noqa: E402


@register_model("parsimony")
class ParsimonyLM(TemplateLM):
    """Wraps a trained Parsimony checkpoint for lm-evaluation-harness."""

    def __init__(
        self,
        checkpoint: str = "runs/flagship/ckpt.pt",
        tokenizer: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device)

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Parsimony checkpoint not found at '{checkpoint}'. Pass "
                f"checkpoint=<path to runs/.../ckpt.pt>."
            )
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.cfg = ParsimonyConfig(**ck["model_cfg"])
        self.model = Parsimony(self.cfg).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

        if tokenizer is None:
            search_roots = [Path("data"), ckpt_path.resolve().parents[2] / "data"]
            candidates = []
            for root in search_roots:
                if root.exists():
                    candidates = sorted(root.glob("tok_*.json"))
                    if candidates:
                        break
            if not candidates:
                raise FileNotFoundError(
                    "Could not find a tokenizer json (looked for data/tok_*.json). "
                    "Pass tokenizer=<path to the .json file> explicitly."
                )
            tokenizer = str(candidates[0])
        self.tokenizer = load_tokenizer(tokenizer)

        self.batch_size = int(batch_size)
        self._max_length = int(max_length) if max_length else int(self.cfg.max_seq_len)

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        """Supports `--model parsimony --model_args checkpoint=...,tokenizer=...`."""
        kwargs = {}
        if arg_string:
            for kv in arg_string.split(","):
                if not kv.strip():
                    continue
                k, v = kv.split("=", 1)
                kwargs[k.strip()] = v.strip()
        kwargs.update(additional_config or {})
        return cls(**kwargs)

    # ---- TemplateLM required interface ------------------------------------

    @property
    def eot_token_id(self) -> int:
        return EOS_ID

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    def tok_encode(
        self, string: str, add_special_tokens: Optional[bool] = None, **kwargs
    ) -> List[int]:
        # Byte-level BPE with no post-processor (src/tokenizer_train.py never
        # attaches one), so encode() never inserts BOS/EOS on its own --
        # `add_special_tokens` has nothing to toggle here.
        return self.tokenizer.encode(string).ids

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def _forward_logprobs(self, batch_ids: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids -> (B, T, V) log-probabilities over the vocab."""
        logits, _ = self.model(batch_ids)
        return F.log_softmax(logits.float(), dim=-1)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[Tuple[float, bool]]:
        # Longest-first so batches are as regular as possible and any
        # length-related failure (e.g. OOM) surfaces early rather than on
        # the last batch of a multi-hour run.
        order = sorted(
            range(len(requests)), key=lambda i: -(len(requests[i][1]) + len(requests[i][2]))
        )
        bs = override_bs or self.batch_size
        out: List[Optional[Tuple[float, bool]]] = [None] * len(requests)

        for start in tqdm(
            range(0, len(order), bs), disable=disable_tqdm, desc="parsimony loglikelihood"
        ):
            batch_idx = order[start : start + bs]
            rows, inplens, contlens, cont_tok_lists = [], [], [], []
            for i in batch_idx:
                _, context_enc, continuation_enc = requests[i]
                # Same convention as lm_eval's own HF backend: the model
                # predicts token t+1 from tokens [0..t], so drop the final
                # token of (context + continuation) from the input -- it's
                # only ever used as a target, never fed in -- and truncate
                # from the left if the pair is longer than the model's
                # context window.
                whole = (context_enc + continuation_enc)[-(self.max_length + 1) :]
                inp = whole[:-1]
                inplen = len(inp)
                contlen = min(len(continuation_enc), inplen)
                rows.append(inp)
                inplens.append(inplen)
                contlens.append(contlen)
                cont_tok_lists.append(continuation_enc[-contlen:] if contlen else [])

            maxlen = max(inplens)
            # Right-pad with <pad>=0 up to a common length so the batch can
            # be a single tensor. This is safe *only* because Parsimony's
            # attention is unconditionally causal
            # (F.scaled_dot_product_attention(..., is_causal=True), no
            # attention_mask input) and every other op (RMSNorm, SwiGLU) is
            # per-position: padding placed after position `inplen - 1` can
            # never influence logits at positions < inplen, so scoring is
            # identical to running that one example alone.
            batch = torch.zeros(len(rows), maxlen, dtype=torch.long, device=self.device)
            for i, r in enumerate(rows):
                if r:
                    batch[i, : len(r)] = torch.tensor(r, dtype=torch.long)

            logp = self._forward_logprobs(batch)  # (B, maxlen, V)

            for i, req_i in enumerate(batch_idx):
                contlen = contlens[i]
                if contlen == 0:
                    # Degenerate case (empty continuation): nothing to score.
                    out[req_i] = (0.0, True)
                    continue
                inplen = inplens[i]
                cont_toks = torch.tensor(cont_tok_lists[i], dtype=torch.long, device=self.device)
                cl = logp[i, inplen - contlen : inplen, :]  # (contlen, V)
                greedy = cl.argmax(dim=-1)
                is_greedy = bool(torch.equal(greedy, cont_toks))
                token_lp = cl.gather(1, cont_toks.unsqueeze(-1)).squeeze(-1)
                out[req_i] = (float(token_lp.sum().item()), is_greedy)

        assert all(o is not None for o in out)
        return out  # type: ignore[return-value]

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        # Token-for-token the same windowing lm_eval's own HFLM uses, via
        # the harness's own helpers -- so a loglikelihood_rolling task
        # (wikitext / wikitext103) gets identical treatment to a native
        # harness model, and the resulting perplexity numbers are directly
        # comparable to ones anyone else reports with this same harness.
        all_windows: List[Tuple[int, tuple]] = []
        counts: List[int] = []
        for req_idx, (string,) in enumerate(
            tqdm([r.args for r in requests], disable=disable_tqdm, desc="parsimony rolling windows")
        ):
            disjoint = [
                lm_utils.make_disjoint_window(w)
                for w in lm_utils.get_rolling_token_windows(
                    token_list=self.tok_encode(string),
                    prefix_token=self.eot_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                )
            ]
            windows = [(None,) + w for w in disjoint]
            all_windows.extend((req_idx, w) for w in windows)
            counts.append(len(windows))

        flat = [w for _, w in all_windows]
        nlls = self._loglikelihood_tokens(flat, disable_tqdm=True) if flat else []

        out, pos = [], 0
        for n in counts:
            out.append(sum(lp for lp, _ in nlls[pos : pos + n]))
            pos += n
        return out

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        out = []
        for context, gen_kwargs in tqdm(
            [r.args for r in requests], disable=disable_tqdm, desc="parsimony generate_until"
        ):
            gen_kwargs = dict(gen_kwargs or {})
            until = gen_kwargs.pop("until", None) or []
            if isinstance(until, str):
                until = [until]
            max_new = int(gen_kwargs.pop("max_gen_toks", self.max_gen_toks))
            temperature = float(gen_kwargs.pop("temperature", 0.0) or 0.0)
            # temperature 0 ("greedy") has no meaning for the multinomial
            # sampler in Parsimony.generate -- force it deterministic with
            # top_k=1 instead, which always leaves exactly one candidate.
            top_k = 1 if temperature <= 0.0 else int(gen_kwargs.pop("top_k", 50))

            ctx_ids = self.tok_encode(context)[-self.max_length :]
            idx = torch.tensor([ctx_ids], dtype=torch.long, device=self.device)
            start_len = idx.shape[1]

            generated = self.model.generate(
                idx,
                max_new_tokens=max_new,
                temperature=max(temperature, 1e-5),
                top_k=top_k,
                eos_id=self.eot_token_id,
            )
            text = self.tok_decode(generated[0, start_len:].tolist())
            for stop in until:
                if stop and stop in text:
                    text = text[: text.index(stop)]
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)
            out.append(text)
        return out
