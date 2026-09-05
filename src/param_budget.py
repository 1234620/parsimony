"""
Parameter budget allocation under a hard cap.

The GIBC Track-01 rule that shapes everything: the 50M parameter cap
*includes token embeddings and the output head*. That turns vocabulary
size from a preprocessing detail into a capacity-allocation decision:
every parameter spent on the embedding table is a parameter not spent
on depth or width.

This module solves, for a given vocabulary size and budget, the largest
transformer that fits -- and reports exactly where the budget went.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelShape:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    embed_rank: Optional[int] = None  # None = standard embedding; int = factorized (V x r) @ (r x d)
    tie_embeddings: bool = True

    # ---- parameter accounting -------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def embedding_params(self) -> int:
        if self.embed_rank is None:
            p = self.vocab_size * self.d_model
        else:
            p = self.vocab_size * self.embed_rank + self.embed_rank * self.d_model
        if not self.tie_embeddings:
            p += self.vocab_size * self.d_model
        return p

    def attention_params_per_layer(self) -> int:
        d, hd = self.d_model, self.head_dim
        q = d * (self.n_heads * hd)
        k = d * (self.n_kv_heads * hd)
        v = d * (self.n_kv_heads * hd)
        o = (self.n_heads * hd) * d
        return q + k + v + o

    def ffn_params_per_layer(self) -> int:
        # SwiGLU: gate (d->f), up (d->f), down (f->d)
        return 3 * self.d_model * self.d_ff

    def norm_params_per_layer(self) -> int:
        return 2 * self.d_model  # pre-attn + pre-ffn RMSNorm gains

    def block_params(self) -> int:
        return (
            self.attention_params_per_layer()
            + self.ffn_params_per_layer()
            + self.norm_params_per_layer()
        )

    def total_params(self) -> int:
        return self.embedding_params() + self.n_layers * self.block_params() + self.d_model

    def breakdown(self) -> dict:
        emb = self.embedding_params()
        blocks = self.n_layers * self.block_params()
        total = self.total_params()
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "d_ff": self.d_ff,
            "embed_rank": self.embed_rank,
            "embedding_params": emb,
            "transformer_params": blocks,
            "total_params": total,
            "embedding_fraction": emb / total,
            "params_per_layer": self.block_params(),
        }

    def summary(self) -> str:
        b = self.breakdown()
        return (
            f"vocab={b['vocab_size']:>6,}  d={b['d_model']:>4}  L={b['n_layers']:>2}  "
            f"heads={b['n_heads']}/{b['n_kv_heads']}kv  d_ff={b['d_ff']:>5}  "
            f"| embed {b['embedding_params']/1e6:>5.2f}M ({b['embedding_fraction']:>5.1%})  "
            f"blocks {b['transformer_params']/1e6:>5.2f}M  "
            f"| total {b['total_params']/1e6:>5.2f}M"
        )


def _round_to(x: int, multiple: int) -> int:
    return max(multiple, int(round(x / multiple)) * multiple)


def solve_shape(
    vocab_size: int,
    budget: int = 50_000_000,
    d_model: int = 512,
    n_heads: Optional[int] = None,
    kv_groups: Optional[int] = None,
    ffn_mult: float = 8 / 3,
    embed_rank: Optional[int] = None,
    min_layers: int = 2,
) -> ModelShape:
    """Largest depth that fits `budget` at the given width and vocabulary."""
    n_heads = n_heads or max(1, d_model // 64)
    kv_groups = kv_groups or n_heads
    d_ff = _round_to(int(d_model * ffn_mult), 64)

    probe = ModelShape(vocab_size, d_model, 1, n_heads, kv_groups, d_ff, embed_rank)
    emb = probe.embedding_params()
    per_layer = probe.block_params()
    remaining = budget - emb - d_model
    n_layers = max(min_layers, remaining // per_layer)
    return ModelShape(vocab_size, d_model, int(n_layers), n_heads, kv_groups, d_ff, embed_rank)


def solve_isoparam_frontier(
    vocab_sizes, budget: int = 50_000_000, d_model: int = 512, **kw
):
    """One ~equal-parameter model per vocabulary size -- the ablation grid."""
    return [solve_shape(v, budget=budget, d_model=d_model, **kw) for v in vocab_sizes]


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser(description="Parameter budget allocator / verifier")
    ap.add_argument("--budget", type=int, default=50_000_000)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=None, help="single vocab size")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.vocab:
        shape = solve_shape(args.vocab, budget=args.budget, d_model=args.d_model)
        print(json.dumps(shape.breakdown(), indent=2) if args.json else shape.summary())
    else:
        print(f"Budget: {args.budget/1e6:.0f}M parameters (embeddings INCLUDED), d_model={args.d_model}\n")
        for s in solve_isoparam_frontier(
            [4096, 8192, 16384, 32768, 50257], budget=args.budget, d_model=args.d_model
        ):
            print("  " + s.summary())
