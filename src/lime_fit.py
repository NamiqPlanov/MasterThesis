"""
LIME attributions with a per-instance locality kernel, shared by both datasets.

Why this exists as a module rather than two copies of a loop: the supervisor's
October review asked for consistent experimental conditions across datasets,
and the condition that has to hold is not that both runs pass the same
kernel_width but that LIME's local surrogate actually fits the model it is
approximating. Those are different requirements, and at 390 features they
point in opposite directions.

LIME weights each perturbation by exp(-D^2 / w^2), with D a distance in feature
space and w defaulting to sqrt(d) * 0.75. The default's adequacy depends on how
flat the model's response surface is near the instance, which varies sharply
within a single alert queue. Measured on IEEE-CIS XGBoost, where predicted
probabilities differ only in the fifth decimal:

    queue position     median local R^2 at 1x / 2x / 3x / 4x
    0-11   (p .99996)      0.000   0.994   0.987   0.915
    120-131 (p .99963)     0.932   0.891   0.768   0.640
    200-211 (p .99944)     0.972   0.644   0.587   0.507

At the head of the queue a narrow kernel draws only perturbations whose model
output is identical, so the weighted target has no variance and the ridge
returns noise -- coefficients of order 1e-29 at R^2 = 0.0002. Further down the
surface has gradient and the narrow kernel is correct, while widening it
destroys locality instead.

A single width per dataset is therefore the wrong structure. The width is
chosen per instance by maximising the surrogate's local R^2, a criterion that
depends only on the LIME fit and never consults whatever the attribution is
later compared against, so it cannot inflate a downstream agreement figure.

On ULB (d = 30) this procedure mostly selects the default, which is the point:
the same code adapts to both datasets rather than being tuned for one.
"""

from __future__ import annotations

import numpy as np
from lime.lime_tabular import LimeTabularExplainer

__all__ = ["default_kernel_width", "build_explainers", "best_fit_attribution"]

# Beyond 4x the surrogate stops being local and R^2 falls on every segment
# measured, so the search range is deliberately short.
KERNEL_MULTS = (1, 2, 3, 4)


def default_kernel_width(d: int) -> float:
    """LIME's own default, sqrt(d) * 0.75, stated explicitly so it can be scaled."""
    return float(np.sqrt(d) * 0.75)


def build_explainers(training_data, feature_names, seed,
                     mults=KERNEL_MULTS):
    """One explainer per candidate width, built once and reused per instance.

    Rebuilding these inside the instance loop dominates the runtime, because
    each one recomputes column statistics over the whole training matrix.
    """
    base = default_kernel_width(len(feature_names))
    return {m: LimeTabularExplainer(
        training_data=training_data, feature_names=list(feature_names),
        class_names=["legit", "fraud"], discretize_continuous=False,
        random_state=seed, mode="classification",
        kernel_width=base * m) for m in mults}


def best_fit_attribution(explainers, x, predict_proba, n_samples):
    """Attribution from whichever candidate width fits this instance best.

    Returns (attribution, local_r2, chosen_multiplier, per_width_r2).
    `local_r2` is the surrogate's own coefficient of determination; a caller
    that aggregates attributions should drop the ones where it is near zero,
    since an attribution from a surrogate that explains none of the local
    behaviour is not evidence about the model.
    """
    d = len(x)
    fits, r2_by_width = {}, {}
    for mult, ex in explainers.items():
        e = ex.explain_instance(x, predict_proba, num_features=d,
                                num_samples=n_samples)
        v = np.zeros(d)
        for fidx, w in e.as_map()[1]:
            v[fidx] = w
        fits[mult] = (v, float(e.score))
        r2_by_width[mult] = float(e.score)
    best = max(fits, key=lambda k: fits[k][1])
    v, r2 = fits[best]
    return v, r2, best, r2_by_width
