"""
Direct measures of explanation quality (Secondary Objective 2).

The literature criticised in Progress Report 2 reports model accuracy and then
presents explanations without measuring them. These functions measure the
explanations themselves:

  fidelity   -- does the attribution actually track the model's behaviour?
  stability  -- does a tiny change in the input produce a different story?
  agreement  -- do two explainers of the same prediction say the same thing?

Each takes an attribution vector (one weight per feature) and returns a scalar,
so results are comparable across explainers, models and datasets.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

__all__ = [
    "deletion_fidelity",
    "deletion_curve", "insertion_curve", "faithfulness",
    "max_sensitivity", "rank_agreement", "jaccard_at_k",
    "ground_truth_recovery", "bootstrap_ci",
]


# --------------------------------------------------------------- fidelity ---
def _curve(predict, x, attribution, baseline, *, insert, k=None):
    """Model output as features are removed (or inserted) most-important first.

    Removal replaces a feature with its baseline value (the dataset mean), so
    the model sees "no information" for it. If the attribution is faithful,
    deleting the features it calls important should move the output fast.
    """
    order = np.argsort(-np.abs(attribution))
    # Walking all d features costs d+1 model evaluations per curve, which is
    # fine at d=30 and impractical at d=390. The informative part of a
    # deletion/insertion curve is its head, so k truncates it to the top-k
    # features. k must be the same across datasets for the numbers to compare.
    if k is not None:
        order = order[:k]
    n = len(order)
    # Build every step of the curve first and score them in ONE call: a
    # per-step predict costs 62 model invocations per instance, which makes
    # 1,000 instances impractical on an ensemble.
    rows = np.empty((n + 1, len(x)), dtype=float)
    current = np.array(baseline, dtype=float) if insert else np.array(x, dtype=float)
    rows[0] = current
    for step, idx in enumerate(order, start=1):
        current = current.copy()
        current[idx] = x[idx] if insert else baseline[idx]
        rows[step] = current
    return np.asarray(predict(rows), dtype=float), n


def deletion_curve(predict, x, attribution, baseline, k=None):
    return _curve(predict, x, attribution, baseline, insert=False, k=k)[0]


def insertion_curve(predict, x, attribution, baseline, k=None):
    return _curve(predict, x, attribution, baseline, insert=True, k=k)[0]


def deletion_fidelity(predict, x, attribution, baseline, k=20):
    """How far the model's output falls as its top-k features are removed.

    Normalised by the starting prediction, so it is comparable across datasets
    of different dimensionality. 1.0 means the output collapsed to zero once
    the named features were taken away; 0.0 means removing them changed
    nothing, i.e. the attribution named features the model was not using.

    Preferred over `faithfulness` below whenever k is much smaller than the
    number of features: an insertion curve that fills only k of d features
    never approaches the real prediction, which drives insertion-minus-deletion
    negative for reasons that have nothing to do with explanation quality.
    """
    curve = deletion_curve(predict, x, attribution, baseline, k)
    start = curve[0]
    if abs(start) < 1e-9:
        return float("nan")          # nothing to explain at this instance
    return float((start - curve.mean()) / abs(start))


def faithfulness(predict, x, attribution, baseline, k=None):
    """insertion AUC - deletion AUC, normalised to the number of steps.

    Higher is better. A random attribution scores near zero because inserting
    and deleting in a meaningless order move the output at the same rate.
    """
    dele = deletion_curve(predict, x, attribution, baseline, k)
    ins = insertion_curve(predict, x, attribution, baseline, k)
    return float(np.trapezoid(ins) - np.trapezoid(dele)) / len(dele)


# -------------------------------------------------------------- stability ---
def max_sensitivity(explain_batch, x, eps=0.05, n_samples=15, rng=None):
    """Largest relative change in the attribution within an eps-ball of x.

    Lower is better. An explanation that flips when the input is nudged by an
    imperceptible amount cannot be the basis of an investigator's decision.

    `explain_batch` takes a (n, d) array and returns (n, d) attributions. The
    perturbations are explained in ONE call: explaining them one at a time
    rebuilds the explainer per row, which is what made this intractable on a
    390-feature forest.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    noise = rng.uniform(-eps, eps, size=(n_samples, x.size)) * (np.abs(x) + 1e-8)
    batch = np.vstack([x.reshape(1, -1), x.reshape(1, -1) + noise])
    attrs = np.asarray(explain_batch(batch), dtype=float)
    base, perturbed = attrs[0], attrs[1:]
    denom = np.linalg.norm(base) or 1.0
    return float(np.max(np.linalg.norm(perturbed - base, axis=1) / denom))


# -------------------------------------------------------------- agreement ---
def rank_agreement(a, b):
    """Spearman correlation between two attribution vectors (by magnitude)."""
    a, b = np.abs(np.asarray(a, float)), np.abs(np.asarray(b, float))
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def jaccard_at_k(a, b, k=5):
    """Overlap of the top-k features two explainers consider most important."""
    top_a = set(np.argsort(-np.abs(np.asarray(a, float)))[:k])
    top_b = set(np.argsort(-np.abs(np.asarray(b, float)))[:k])
    return len(top_a & top_b) / len(top_a | top_b)


# ------------------------------------------------- synthetic ground truth ---
def ground_truth_recovery(attribution, true_importance, k=5):
    """Only usable where the true importances are known by construction.

    This is the control: a metric or explainer that cannot recover a known
    answer cannot be trusted on an unknown one.
    """
    return {
        "spearman_vs_truth": rank_agreement(attribution, true_importance),
        "jaccard_at_k": jaccard_at_k(attribution, true_importance, k),
        "noise_mass": float(
            np.abs(np.asarray(attribution, float))[np.asarray(true_importance) == 0].sum()
            / (np.abs(np.asarray(attribution, float)).sum() + 1e-12)
        ),
    }


# ------------------------------------------------------------------ stats ---
def bootstrap_ci(values, n_boot=2000, alpha=0.05, rng=None):
    """Percentile bootstrap CI for the mean; reported with every metric."""
    rng = rng or np.random.default_rng(0)
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return (float("nan"),) * 3
    means = [rng.choice(values, values.size, replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)
