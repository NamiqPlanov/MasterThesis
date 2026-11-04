"""
SHAP vs LIME on IEEE-CIS -- closing the last consistency gap in S2.

03_* measures cross-explainer agreement on ULB; 06_* did not, because LIME on
390 features was left out of the first pass. The supervisor's review asked for
consistent experimental conditions across the two datasets, and a metric that
exists on one dataset and not the other is the clearest violation of that.

WHY THIS SCRIPT TUNES LIME'S KERNEL
-----------------------------------
Copying ULB's LIME settings verbatim produces nonsense here, and the reason is
instructive enough to belong in the thesis. LIME weights each perturbation by
exp(-D^2 / kernel_width^2) with D measured in feature space, and its default
width is sqrt(d)*0.75. At d = 390 on an alert whose score is ~0.99997, every
perturbation receives effectively zero weight, the ridge has nothing to fit,
and the returned coefficients are numerical noise -- magnitudes of 1e-29, with
a local R^2 of 0.0002. A rank correlation computed against that noise is
meaningless, and reporting it would reproduce exactly the unverified-metric
practice this thesis criticises.

More samples do not fix it (R^2 stays ~0.001 at 10,400 samples). Widening the
kernel does: at 2x the default, median R^2 at the head of the queue is 0.99.

Crucially, the width that works is not a property of the dataset but of the
instance. Measured along the alert queue, with predicted probabilities that
differ only in the fifth decimal:

    queue position     median local R^2 at 1x / 2x / 3x / 4x
    0-11   (p .99996)      0.000   0.994   0.987   0.915
    120-131 (p .99963)     0.932   0.891   0.768   0.640
    200-211 (p .99944)     0.972   0.644   0.587   0.507

At the very head of the queue the model sits on a flat plateau, so a narrow
kernel draws only points whose output is identical, the weighted target has no
variance, and the ridge returns noise. Further down the surface has gradient,
the narrow kernel works, and widening it destroys locality instead.

A single width per dataset is therefore the wrong structure, and one chosen on
held-out instances is actively misleading -- an earlier version of this script
selected 1x on positions 200-259 and then reported a median R^2 of 0.010 on
positions 0-119. The width is consequently chosen PER INSTANCE by maximising
the local surrogate's R^2, which is a property of the LIME fit alone and never
consults the SHAP vector it is later compared against. Instances whose best
surrogate still fails to fit (R^2 below MIN_R2) are excluded and counted,
because a rank correlation against a surrogate that explains none of the local
variance measures nothing.

So the condition held constant across the two datasets is not the literal
kernel_width formula but the requirement that LIME's local surrogate actually
fits. ULB satisfies it at the default width (median R^2 0.82-0.96); IEEE-CIS
needs the search.

Sample count is 2,000 rather than ULB's 800: fitting 390 coefficients from 800
draws is a 2:1 ratio where ULB enjoys 27:1. Matching the ratio exactly would
need 10,400 draws, which costs 13x for no measured gain in R^2.

One further deviation. XGBoost is fitted on the raw matrix and consumes NaN
natively, but LIME derives its perturbation distribution from column means and
standard deviations, which are undefined on a matrix that is 41% missing.
XGBoost is therefore explained here on the median-imputed view used for
RandomForest. SHAP and LIME see the identical vector, so the agreement
measurement stays valid; it is the input view that differs from 06_*.

    .venv\\Scripts\\python.exe -u experiments\\13_lime_ieee.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from xai_metrics import bootstrap_ci, jaccard_at_k, rank_agreement  # noqa: E402

warnings.filterwarnings("ignore")
# see the note in 06_*: filterwarnings binds this process only, and joblib's
# workers are separate processes that re-emit everything to the shared stderr
os.environ.setdefault("PYTHONWARNINGS", "ignore")

ART, OUT = ROOT / "artifacts", ROOT / "results"
SEED = 42
N_ALERT, N_RANDOM = 400, 600      # only to reproduce 06_*'s instance ordering
N_LIME = 120                      # identical to 03_* on ULB
LIME_SAMPLES = 2000
TOP_K = 5

# kernel-width multipliers searched per instance, as a factor on LIME's own
# sqrt(d)*0.75. The profile above shows the useful range is 1x-4x: beyond that
# the surrogate stops being local and R^2 falls on every segment measured.
KERNEL_MULTS = (1, 2, 3, 4)
# an attribution from a surrogate that explains less than half the local
# variance is not evidence about the model, so it is excluded and counted
MIN_R2 = 0.5


def build_lime(Xm, cols, kernel_width):
    return LimeTabularExplainer(
        training_data=Xm, feature_names=cols,
        class_names=["legit", "fraud"], discretize_continuous=False,
        random_state=SEED, mode="classification", kernel_width=kernel_width)


def lime_vector(explainer, x, predict, d, n_samples):
    """LIME attribution for one instance, plus the surrogate's local R^2."""
    e = explainer.explain_instance(x, predict, num_features=d,
                                   num_samples=n_samples)
    v = np.zeros(d)
    for fidx, w in e.as_map()[1]:
        v[fidx] = w
    return v, float(e.score)


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    split = joblib.load(ART / "ieee_split.joblib")
    cols = split["columns"]
    yte = split["yte"]
    Xte_raw, Xte_s = split["Xte_raw"], split["Xte_scaled"]
    d = len(cols)
    base_kw = float(np.sqrt(d) * 0.75)

    print("=" * 78)
    print("  SHAP vs LIME ON IEEE-CIS")
    print("=" * 78)
    print(f"  features: {d}  |  instances: {N_LIME}  "
          f"|  LIME samples/instance: {LIME_SAMPLES}")
    print(f"  LIME default kernel_width = sqrt(d)*0.75 = {base_kw:.2f}",
          flush=True)

    models = {n: joblib.load(ART / f"ieee_{n}.joblib")
              for n in ("LogisticRegression", "RandomForest", "XGBoost")}

    med = split["median"]
    Xte_imp = np.where(np.isnan(Xte_raw), med, Xte_raw).astype(np.float32)
    # XGBoost deviates from 06_* here; see the module docstring
    inputs = {"LogisticRegression": Xte_s, "RandomForest": Xte_imp,
              "XGBoost": Xte_imp}

    # reproduce 06_*'s instance ordering exactly: the alert queue is ranked by
    # XGBoost on the RAW matrix, as it is there, so the same rows are explained
    ref = models["XGBoost"].predict_proba(Xte_raw)[:, 1]
    alert_idx = np.argsort(-ref)[:N_ALERT]
    pool = np.setdiff1d(np.arange(len(yte)), alert_idx)
    rand_idx = rng.choice(pool, N_RANDOM, replace=False)
    idx_all = np.concatenate([alert_idx, rand_idx])
    idx = idx_all[:N_LIME]
    print(f"  reporting on the first {N_LIME} of the alert queue "
          f"({int(yte[idx].sum())} of them fraud)")
    print(f"  kernel width chosen per instance from "
          f"{[f'{m}x' for m in KERNEL_MULTS]} by local R2, "
          f"floor R2 >= {MIN_R2}\n", flush=True)

    report = {}
    for name, model in models.items():
        t = time.time()
        Xm = inputs[name]

        if name == "LogisticRegression":
            ex = shap.LinearExplainer(
                model, shap.maskers.Independent(Xm, max_samples=2000))
        else:
            ex = shap.TreeExplainer(model)
        sv = np.asarray(ex.shap_values(Xm[idx]))
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        # one explainer per candidate width, reused across instances
        explainers = {m: build_lime(Xm, cols, base_kw * m)
                      for m in KERNEL_MULTS}

        rhos, jacs, best_r2, chosen = [], [], [], []
        naive_rhos, naive_r2 = [], []     # what ULB's settings would have given
        for i in range(N_LIME):
            fits = {}
            for m in KERNEL_MULTS:
                lv, r2 = lime_vector(explainers[m], Xm[idx[i]],
                                     model.predict_proba, d, LIME_SAMPLES)
                fits[m] = (lv, r2)
            naive_rhos.append(rank_agreement(sv[i], fits[1][0]))
            naive_r2.append(fits[1][1])

            m_best = max(fits, key=lambda k: fits[k][1])
            lv, r2 = fits[m_best]
            chosen.append(m_best)
            best_r2.append(r2)
            if r2 >= MIN_R2:
                rhos.append(rank_agreement(sv[i], lv))
                jacs.append(jaccard_at_k(sv[i], lv, TOP_K))
            if (i + 1) % 30 == 0:
                print(f"    {name}: {i + 1}/{N_LIME} "
                      f"({time.time() - t:.0f}s)", flush=True)

        ar, aj = bootstrap_ci(rhos), bootstrap_ci(jacs)
        n_fit = len(rhos)
        n_ok = int(np.isfinite(rhos).sum())
        hist = {f"{m}x": int(np.sum(np.array(chosen) == m))
                for m in KERNEL_MULTS}
        print(f"\n  --- {name}  ({time.time() - t:.0f}s) ---")
        print(f"  {'SHAP vs LIME, Spearman':<34}{ar[0]:>8.3f}  "
              f"[{ar[1]:.3f}, {ar[2]:.3f}]")
        print(f"  {'SHAP vs LIME, Jaccard@' + str(TOP_K):<34}{aj[0]:>8.3f}  "
              f"[{aj[1]:.3f}, {aj[2]:.3f}]")
        print(f"  {'LIME local R2 (median, tuned)':<34}"
              f"{np.median(best_r2):>8.3f}")
        print(f"  {'LIME local R2 (median, at 1x)':<34}"
              f"{np.median(naive_r2):>8.3f}")
        print(f"  {'instances meeting the R2 floor':<34}{n_fit:>8d} of {N_LIME}")
        print(f"  {'of those, rho defined':<34}{n_ok:>8d}")
        print(f"  widths chosen: " + ", ".join(
            f"{k} {v}" for k, v in hist.items()) + "\n", flush=True)

        report[name] = {"shap_vs_lime_spearman": ar,
                        "shap_vs_lime_jaccard": aj,
                        "lime_local_r2_median": float(np.median(best_r2)),
                        "lime_local_r2_median_at_default_width":
                            float(np.median(naive_r2)),
                        "kernel_multiplier_counts": hist,
                        "n_meeting_r2_floor": n_fit,
                        "n_defined": n_ok, "n_total": N_LIME}

    (OUT / "s2_shap_vs_lime_ieee.json").write_text(
        json.dumps({"seed": SEED, "n_lime": N_LIME,
                    "lime_samples": LIME_SAMPLES, "top_k": TOP_K,
                    "n_features": d,
                    "lime_default_kernel_width": base_kw,
                    "kernel_multipliers_searched": list(KERNEL_MULTS),
                    "kernel_selected_per_instance": True,
                    "min_local_r2": MIN_R2,
                    "xgboost_input_view": "median-imputed (LIME cannot derive "
                                          "perturbation statistics from NaN)",
                    "per_model": report},
                   indent=2, default=float), encoding="utf-8")
    print(f"  wrote results/s2_shap_vs_lime_ieee.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
