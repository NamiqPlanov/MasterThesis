"""
Why do ULB and IEEE-CIS fail in opposite directions at the same alert budget?

The supervisor asked for a technical explanation rather than a restatement of
the difference. Four candidate causes are examined: the arithmetic of the
budget itself, probability calibration, threshold sensitivity, and how well
each model separates fraud from legitimate traffic.

The first turns out to dominate. precision@k and recall@k are both bounded by
the ratio of review capacity to fraud count, and because k is fixed both
collapse to the same normalised quantity, caught / min(k, F) -- the share of
attainable catches the model actually makes. Reported that way the datasets
stop disagreeing.

    .venv\\Scripts\\python.exe -u experiments\\15_cross_dataset_diagnosis.py
"""

from __future__ import annotations
import json,os,sys,warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

warnings.filterwarnings('ignore')
os.environ.setdefault('PYTHONWARNINGS','ignore')

ROOT = Path(__file__).resolve().parent.parent
ART,OUT,DATA = ROOT / 'artifacts',ROOT/'results',ROOT/'data'
BUDGETS = (0.001,0.005,0.01)
# A finer sweep for the threshold-sensitivity question. Three points cannot
# show where a dataset's attainment curve bends, and the bend is the useful
# output: below k/F = 1 recall is capacity-bound, above it precision is.
SWEEP = (0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005,
         0.0075, 0.01, 0.02, 0.03, 0.05)
MODELS = ('LogisticRegression','RandomForest','XGBoost')
N_BINS = 15
SEED = 42

def scores_for(tag):
    """Test-set scores per model, each fed the representation it was fitted on."""
    split = joblib.load(ART / f"{tag}_split.joblib")
    yte = split["yte"]
    if tag == "ulb":
        views = {m: split["Xte"] for m in MODELS}
        amount = split["Xte"][:, split["columns"].index("Amount")]
    else:
        raw, med = split["Xte_raw"], split["median"]
        imp = np.where(np.isnan(raw), med, raw).astype(np.float32)
        views = {"LogisticRegression": split["Xte_scaled"],
                 "RandomForest": imp, "XGBoost": raw}
        amount = imp[:, split["columns"].index("TransactionAmt")]
    out = {}
    for m in MODELS:
        mdl = joblib.load(ART / f"{tag}_{m}.joblib")
        out[m] = mdl.predict_proba(views[m])[:, 1]
    return yte, out, amount


def attainment(y, s, budget):
    """Precision and recall at a budget, plus the ceiling each can reach.

    With k alerts and F frauds, at most min(k, F) frauds can be caught, so
    precision <= min(1, F/k) and recall <= min(1, k/F). Dividing each by its
    ceiling gives the same number -- that is the point.
    """
    n = len(s)
    k = max(1, int(n * budget))
    caught = int(y[np.argsort(-s)[:k]].sum())
    F = int(y.sum())
    best = min(k, F)
    return {"k": k, "frauds": F, "k_over_F": k / F,
            "caught": caught,
            "precision": caught / k, "recall": caught / F,
            "max_precision": min(1.0, F / k), "max_recall": min(1.0, k / F),
            "attainment": caught / best if best else float("nan"),
            # what a team would get with no model at all, and with one cheap rule
            "random_expected_catches": k * F / n}


def calibration(y, s):
    """Brier, expected calibration error, and the calibration curve's slope.

    A slope below 1 means the scores are over-confident at the extremes, which
    is what undersampling is known to cause (Dal Pozzolo et al., 2015) and a
    plausible reason two datasets behave differently at the same budget.
    """
    p = np.clip(np.asarray(s, dtype=np.float64), 1e-12, 1 - 1e-12)    
    edges = np.quantile(p, np.linspace(0, 1, N_BINS + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    ece, curve = 0.0, []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        conf, freq = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - freq)
        curve.append({"bin": b, "n": int(m.sum()),
                      "mean_predicted": conf, "observed_rate": freq})
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(max_iter=2000).fit(logit, y)
    return {"brier": float(brier_score_loss(y, p)),
            "ece": float(ece),
            "calibration_slope": float(lr.coef_[0][0]),
            "calibration_intercept": float(lr.intercept_[0]),
            "curve": curve}


def separation(y, s):
    """How far apart the two score distributions are, independent of threshold."""
    pos, neg = s[y == 1], s[y == 0]
    return {"ks_statistic": float(ks_2samp(pos, neg).statistic),
            "median_score_fraud": float(np.median(pos)),
            "median_score_legit": float(np.median(neg)),
            "p90_score_legit": float(np.quantile(neg, 0.90)),
            "frauds_above_legit_p99": float((pos > np.quantile(neg, 0.99)).mean())}


def main() -> int:
    report = {"budgets": list(BUDGETS), "n_bins": N_BINS, "datasets": {}}
    for tag, label in (("ulb", "ULB"), ("ieee", "IEEE-CIS")):
        yte, scores, amount = scores_for(tag)
        n, F = len(yte), int(yte.sum())
        print("=" * 78)
        print(f"  {label}: {n:,} test rows, {F:,} frauds "
              f"({F / n * 100:.3f}%)")
        print("=" * 78)

        ds = {"n_test": n, "frauds": F, "base_rate": F / n,
              "per_model": {}, "baselines": {}}

        # a cheap rule and a coin, at the same budget, for the comparison the
        # supervisor asked for: is the model a real improvement under the
        # constraint a fraud team actually works under?
        for name, s in (("rank_by_amount", amount),
                        ("random", np.random.default_rng(SEED).random(n))):
            ds["baselines"][name] = {f"b={b}": attainment(yte, s, b)
                                     for b in BUDGETS}

        for m in MODELS:
            s = scores[m]
            ds["per_model"][m] = {
                "budget": {f"b={b}": attainment(yte, s, b) for b in BUDGETS},
                "sweep": {f"b={b}": attainment(yte, s, b) for b in SWEEP},
                "calibration": calibration(yte, s),
                "separation": separation(yte, s)}
            a = ds["per_model"][m]["budget"]["b=0.005"]
            c = ds["per_model"][m]["calibration"]
            print(f"  {m:<20} k/F={a['k_over_F']:5.2f}  "
                  f"prec {a['precision']:.3f}/{a['max_precision']:.3f}  "
                  f"rec {a['recall']:.3f}/{a['max_recall']:.3f}  "
                  f"attainment {a['attainment'] * 100:5.1f}%  "
                  f"slope {c['calibration_slope']:.2f}  ECE {c['ece']:.4f}")

        base = ds["baselines"]["rank_by_amount"]["b=0.005"]["caught"]
        best = max(ds["per_model"][m]["budget"]["b=0.005"]["caught"]
                   for m in MODELS)
        rnd = ds["baselines"]["random"]["b=0.005"]["random_expected_catches"]
        print(f"\n  catches at b=0.5%: best model {best}, "
              f"amount rule {base}, random {rnd:.1f}  "
              f"-> lift over random x{best / max(rnd, 1e-9):.1f}")

        # --- threshold sensitivity --------------------------------------
        # Which constraint binds, and where does it switch? Below k/F = 1 the
        # budget cannot hold the fraud present, so recall is capped at k/F
        # however good the model is; above it the budget exceeds the fraud and
        # precision is capped at F/k instead. The crossing is the operating
        # point where capacity and fraud volume are matched.
        best_m = max(MODELS, key=lambda mm:
                     ds["per_model"][mm]["budget"]["b=0.005"]["attainment"])
        print(f"\n  threshold sensitivity, {best_m} "
              f"(budget sweep, {len(SWEEP)} points):")
        print(f"    {'budget':>8}{'alerts':>8}{'k/F':>7}{'prec':>8}{'rec':>8}"
              f"{'attain':>9}   binding constraint")
        prev_ratio = None
        crossing = None
        for b in SWEEP:
            a = ds["per_model"][best_m]["sweep"][f"b={b}"]
            binding = "recall (capacity)" if a["k_over_F"] < 1 else "precision"
            mark = ""
            if prev_ratio is not None and prev_ratio < 1 <= a["k_over_F"]:
                mark, crossing = "   <- capacity meets fraud volume", b
            print(f"    {b * 100:>7.2f}%{a['k']:>8}{a['k_over_F']:>7.2f}"
                  f"{a['precision']:>8.3f}{a['recall']:>8.3f}"
                  f"{a['attainment'] * 100:>8.1f}%   {binding}{mark}")
            prev_ratio = a["k_over_F"]
        ds["binding_constraint_crossing"] = crossing
        ds["attainment_sweep_best_model"] = best_m
        if crossing:
            print(f"    capacity first exceeds the fraud present at "
                  f"b = {crossing * 100:.2f}%; beyond it precision is capped "
                  f"by F/k regardless of model quality")
        else:
            print(f"    capacity never reaches the fraud present within "
                  f"b <= {max(SWEEP) * 100:.0f}%, so recall stays "
                  f"capacity-bound throughout")
        print()
        report["datasets"][label] = ds

    (OUT / "s6_cross_dataset_diagnosis.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")
    print("  wrote results/s6_cross_dataset_diagnosis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())    