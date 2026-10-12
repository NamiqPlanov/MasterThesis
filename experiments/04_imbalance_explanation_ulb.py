"""
Stage S3 (Secondary Objective 3): does handling class imbalance distort the
explanations?

The "explainability-imbalance paradox" claims that resampling creates synthetic
minority points, and that post-hoc explainers then attribute importance partly
to artefacts of that synthetic data rather than to real fraud signal. The claim
currently rests on a single unreviewed preprint, which makes it worth testing
rather than citing.

Design: ONE architecture, ONE test set, three training regimes --
natural imbalance, random undersampling, SMOTE. Predictive performance is
recorded so that any divergence in explanations can be read against it: if
accuracy is unchanged but the explanations move, the explanations are tracking
the resampling rather than the fraud.

H0: SHAP rankings are invariant to the resampling regime.

    .venv\\Scripts\\python.exe experiments\\04_imbalance_explanation_ulb.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from xai_metrics import bootstrap_ci, jaccard_at_k, rank_agreement  # noqa: E402

warnings.filterwarnings("ignore")
CSV = ROOT / "data" / "ulb" / "creditcard.csv"
OUT = ROOT / "results"
SEED = 42
TEST_FRACTION = 0.30
N_EXPLAIN = 400          # the alert queue, where explanations are actually read
TOP_K = 5


def fit_xgb(X, y):
    pos = max(int((y == 1).sum()), 1)
    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
        scale_pos_weight=float((y == 0).sum() / pos),
        n_jobs=-1, random_state=SEED, tree_method="hist").fit(X, y)


def main() -> int:
    t0 = time.time()
    df = pd.read_csv(CSV).sort_values("Time").reset_index(drop=True)
    y = df["Class"].to_numpy()
    X = df.drop(columns=["Class"])
    cut = int(len(df) * (1 - TEST_FRACTION))
    scaler = StandardScaler().fit(X.iloc[:cut])
    Xtr, Xte = scaler.transform(X.iloc[:cut]), scaler.transform(X.iloc[cut:])
    ytr, yte = y[:cut], y[cut:]
    cols = list(X.columns)

    print("=" * 78)
    print("  S3 - DOES RESAMPLING CHANGE THE EXPLANATION?")
    print("=" * 78)
    print(f"  one architecture (XGBoost), one test set ({len(yte):,} rows), "
          "three training regimes")

    regimes = {}
    regimes["natural"] = (Xtr, ytr)
    rus = RandomUnderSampler(random_state=SEED)
    regimes["undersample"] = rus.fit_resample(Xtr, ytr)
    sm = SMOTE(random_state=SEED, k_neighbors=5)
    regimes["smote"] = sm.fit_resample(Xtr, ytr)

    models, perf = {}, {}
    print(f"\n  {'regime':<14}{'train rows':>12}{'fraud %':>10}{'AUC-PR':>10}")
    print("  " + "-" * 46)
    for name, (Xr, yr) in regimes.items():
        m = fit_xgb(Xr, yr)
        ap = average_precision_score(yte, m.predict_proba(Xte)[:, 1])
        models[name], perf[name] = m, ap
        print(f"  {name:<14}{len(yr):>12,}{yr.mean() * 100:>9.1f}%{ap:>10.4f}")

    # the same instances explained under every regime
    ref_scores = models["natural"].predict_proba(Xte)[:, 1]
    idx = np.argsort(-ref_scores)[:N_EXPLAIN]
    sv = {}
    for name, m in models.items():
        v = np.asarray(shap.TreeExplainer(m).shap_values(Xte[idx]))
        sv[name] = v[:, :, 1] if v.ndim == 3 else v

    print(f"\n  --- agreement with the NATURAL-imbalance explanations "
          f"({N_EXPLAIN} alerts) ---")
    print(f"  {'regime':<14}{'Spearman':>20}{'Jaccard@' + str(TOP_K):>22}")
    comp = {}
    for name in ("undersample", "smote"):
        rho = bootstrap_ci([rank_agreement(sv["natural"][i], sv[name][i])
                            for i in range(N_EXPLAIN)])
        jac = bootstrap_ci([jaccard_at_k(sv["natural"][i], sv[name][i], TOP_K)
                            for i in range(N_EXPLAIN)])
        print(f"  {name:<14}{rho[0]:>9.3f} [{rho[1]:.3f}, {rho[2]:.3f}]"
              f"{jac[0]:>11.3f} [{jac[1]:.3f}, {jac[2]:.3f}]")
        comp[name] = {"spearman": rho, "jaccard": jac}

    # H0: rankings invariant. A paired test against perfect agreement (rho = 1).
    print("\n  --- test of H0: rankings invariant to regime ---")
    tests = {}
    for name in ("undersample", "smote"):
        rhos = np.array([rank_agreement(sv["natural"][i], sv[name][i])
                         for i in range(N_EXPLAIN)])
        rhos = rhos[np.isfinite(rhos)]
        stat, p = wilcoxon(rhos - 1.0, alternative="less")
        # effect size: how far from perfect agreement, in SDs
        d = float((1.0 - rhos.mean()) / (rhos.std(ddof=1) or 1e-9))
        print(f"  natural vs {name:<12} Wilcoxon p = {p:.3g}   "
              f"mean rho {rhos.mean():.3f}   Cohen's d {d:.2f}")
        tests[name] = {"p_value": float(p), "mean_rho": float(rhos.mean()),
                       "cohens_d": d, "n": int(len(rhos))}

    # which feature gets blamed, per regime
    print("\n  --- most-named feature in the alert queue, per regime ---")
    named = {}
    for name, v in sv.items():
        top1 = [cols[int(np.argmax(np.abs(v[i])))] for i in range(N_EXPLAIN)]
        vals, counts = np.unique(top1, return_counts=True)
        order = np.argsort(-counts)[:3]
        print(f"  {name:<14}" + ", ".join(
            f"{vals[o]} ({counts[o] * 100 / N_EXPLAIN:.0f}%)" for o in order))
        named[name] = {str(vals[o]): int(counts[o]) for o in order}

    verdict = all(tests[n]["p_value"] < 0.05 for n in tests)
    print("\n" + "=" * 78)
    print(f"  H0 {'REJECTED' if verdict else 'not rejected'}: explanations "
          f"{'change' if verdict else 'do not detectably change'} with the "
          "resampling regime")
    spread = max(perf.values()) - min(perf.values())
    print(f"  meanwhile AUC-PR varies by only {spread:.4f} across the three regimes")
    print("=" * 78)

    OUT.mkdir(exist_ok=True)
    (OUT / "s3_imbalance_explanation_ulb.json").write_text(
        json.dumps({"seed": SEED, "performance": perf, "agreement": comp,
                    "tests": tests, "top_features": named,
                    "h0_rejected": verdict}, indent=2, default=float),
        encoding="utf-8")
    print(f"\n  wrote results/s3_imbalance_explanation_ulb.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
