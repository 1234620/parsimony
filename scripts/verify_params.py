"""Verify analytical parameter accounting against the instantiated model.

GIBC Track-01 requires a printed parameter count + model config. This script
is that artifact, and it doubles as a regression test on the budget solver.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from param_budget import solve_shape
from model import Parsimony, ParsimonyConfig

BUDGET = 50_000_000
print(f"{'vocab':>7} {'d':>5} {'L':>3} | {'analytic':>12} {'actual':>12} {'delta':>6} | {'emb%':>6} | fits {BUDGET/1e6:.0f}M")
print("-" * 78)

ok = True
for vocab in [4096, 8192, 16384, 32768, 50257]:
    shape = solve_shape(vocab, budget=BUDGET, d_model=512)
    cfg = ParsimonyConfig.from_shape(shape, max_seq_len=512)
    model = Parsimony(cfg)
    analytic = shape.total_params()
    actual = model.num_parameters()
    delta = actual - analytic
    frac = shape.breakdown()["embedding_fraction"]
    within = actual <= BUDGET
    ok &= within and abs(delta) < 0.01 * analytic
    print(f"{vocab:>7,} {cfg.d_model:>5} {cfg.n_layers:>3} | {analytic:>12,} {actual:>12,} {delta:>+6,} | {frac:>5.1%} | {'YES' if within else 'NO ':>4}")
    del model

print("-" * 78)
print("accounting consistent and under budget:", ok)

# forward/backward sanity on the flagship config
shape = solve_shape(8192, budget=BUDGET, d_model=512)
cfg = ParsimonyConfig.from_shape(shape, max_seq_len=256)
m = Parsimony(cfg)
x = torch.randint(0, cfg.vocab_size, (2, 65))
# NOTE: targets must be shifted by one. Using targets==inputs lets tied
# embeddings exploit an identity shortcut and reports a misleadingly low loss.
logits, loss = m(x[:, :-1], targets=x[:, 1:])
loss.backward()
gnorm = sum((p.grad**2).sum().item() for p in m.parameters() if p.grad is not None) ** 0.5
print(f"\nflagship forward: logits{tuple(logits.shape)}  loss={loss.item():.4f} "
      f"(ln(V)={torch.log(torch.tensor(float(cfg.vocab_size))).item():.4f} expected at init)")
print(f"backward ok, grad norm={gnorm:.3f}")
