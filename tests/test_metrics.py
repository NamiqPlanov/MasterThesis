"""
Guard rails for the explanation-quality metrics.

Three metric errors reached a long experiment run before being caught, each
costing ~45 minutes. These assertions encode what each measure must do, so the
next error is caught in two seconds instead.

    .venv\\Scripts\\python.exe tests\\test_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from xai_metrics import (  # noqa: E402
    bootstrap_ci, deletion_fidelity, jaccard_at_k, max_sensitivity,
    rank_agreement,
)

FAIL = []


def check(name, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not condition:
        FAIL.append(name)


# a model that uses only feature 0
def predict(a):
    return 1 / (1 + np.exp(-(3 * np.asarray(a)[:, 0])))


base = np.zeros(4)
good = np.array([1.0, 0.0, 0.0, 0.0])
bad = np.array([0.0, 0.0, 0.0, 1.0])

print("deletion_fidelity")
x_alert = np.array([2.0, 1.0, -1.0, 0.5])        # model scores this high
g = deletion_fidelity(predict, x_alert, good, base, k=1)
b = deletion_fidelity(predict, x_alert, bad, base, k=1)
check("correct attribution beats wrong one", g > b, f"{g:.3f} vs {b:.3f}")
check("wrong attribution scores ~0", abs(b) < 1e-6, f"{b:.3g}")
check("stays within [-1, 1]", -1 <= g <= 1, f"{g:.3f}")

# the case that produced -1218: a legitimate instance the model scores near 0
x_legit = np.array([-5.0, 0.2, 0.1, -0.3])
vals = [deletion_fidelity(predict, x_legit, a, base, k=2)
        for a in (good, bad, np.array([0.1, 0.2, 0.3, 0.4]))]
check("near-zero prediction does not explode",
      all(abs(v) <= 1.0 for v in vals), f"max |v| = {max(abs(v) for v in vals):.3g}")

print("\nmax_sensitivity")
# a stable explainer returns the same attribution regardless of input
stable = lambda A: np.tile(good, (len(A), 1))          # noqa: E731
jumpy = lambda A: np.random.default_rng(0).normal(size=(len(A), 4))  # noqa: E731
s_stable = max_sensitivity(stable, x_alert, n_samples=5)
s_jumpy = max_sensitivity(jumpy, x_alert, n_samples=5)
check("stable explainer scores ~0", s_stable < 1e-9, f"{s_stable:.3g}")
check("unstable explainer scores higher", s_jumpy > s_stable, f"{s_jumpy:.2f}")

print("\nagreement")
check("identical vectors agree perfectly",
      abs(rank_agreement(good + 0.1, good + 0.1) - 1.0) < 1e-9)
check("jaccard of identical top-k is 1.0",
      jaccard_at_k(np.array([3, 2, 1, 0]), np.array([30, 20, 10, 0]), 2) == 1.0)
check("jaccard of disjoint top-k is 0.0",
      jaccard_at_k(np.array([3, 2, 0, 0]), np.array([0, 0, 2, 3]), 2) == 0.0)

print("\nbootstrap_ci")
m, lo, hi = bootstrap_ci(np.full(100, 0.5))
check("constant input gives zero-width interval", abs(hi - lo) < 1e-9, f"{m:.3f}")
m, lo, hi = bootstrap_ci(np.concatenate([np.full(50, 0.0), np.full(50, 1.0)]))
check("interval brackets the mean", lo <= m <= hi, f"{m:.2f} [{lo:.2f}, {hi:.2f}]")
check("NaNs are dropped, not propagated",
      np.isfinite(bootstrap_ci([0.1, np.nan, 0.3])[0]))

print()
if FAIL:
    print(f"{len(FAIL)} FAILED: {', '.join(FAIL)}")
    raise SystemExit(1)
print("all metric guard rails pass")
