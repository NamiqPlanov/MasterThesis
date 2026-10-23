"""
Stage S3 replicated on IEEE-CIS: does the imbalance-explanation paradox hold
on a second, very different dataset?

The ULB result (04_*) showed SHAP rankings diverging sharply across resampling
regimes while AUC-PR barely moved. One dataset is a finding; two is a result.
IEEE-CIS is a strong replication test because almost everything differs:

  domain        card-present extract  ->  real e-commerce
  fraud rate    0.17%                ->  3.37%  (20x)
  features      30                   ->  390   (13x)
  span          48 hours             ->  85 days

H0: SHAP rankings are invariant to the resampling regime.

MEMORY: a 150,000-row chronological subsample, because SMOTE on 390 features
roughly doubles the training matrix and the machine has under 2 GB free. Time
ordering is preserved; the deviation is recorded in the results file.

    .venv\\Scripts\\python.exe experiments\\10_imbalance_explanation_ieee.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from xai_metrics import bootstrap_ci, jaccard_at_k, rank_agreement  # noqa: E402

warnings.filterwarnings("ignore")
CSV = ROOT / "data" / "ieee" / "train_transaction.csv"
OUT = ROOT / "results"
SEED = 42
N_ROWS = 150_000
TEST_FRACTION = 0.30
N_EXPLAIN = 400
TOP_K = 5
MAX_MISSING = 0.90


def fit_xgb(X, y):
    pos = max(int((y == 1).sum()), 1)
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
        scale_pos_weight=float((y == 0).sum() / pos),
        n_jobs=-1, random_state=SEED, tree_method="hist").fit(X, y)


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print("  S3 REPLICATION - DOES RESAMPLING CHANGE THE EXPLANATION ON IEEE-CIS?")
    print("=" * 78)

    df = pd.read_csv(CSV, nrows=N_ROWS).sort_values("TransactionDT")
    y = df["isFraud"].to_numpy().astype(np.int8)
    df = df.drop(columns=["isFraud", "TransactionID"])
    for c in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        df[c] = pd.factorize(df[c])[0].astype(np.float32)
    df = df[df.columns[df.isna().mean() <= MAX_MISSING]].astype(np.float32)
    cols = list(df.columns)

    cut = int(len(df) * (1 - TEST_FRACTION))
    X = df.to_numpy(dtype=np.float32)
    del df
    gc.collect()
    med = np.nan_to_num(np.nanmedian(X[:cut], axis=0)).astype(np.float32)
    X = np.where(np.isnan(X), med, X).astype(np.float32)   # SMOTE needs no NaN
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    del X
    gc.collect()

    print(f"  rows {len(ytr) + len(yte):,} ({N_ROWS:,}-row chronological subsample "
          f"of 590,540)   features {len(cols)}")
    print(f"  train {len(ytr):,} ({ytr.mean()*100:.2f}% fraud)   "
          f"test {len(yte):,} ({yte.mean()*100:.2f}% fraud)")

    models, perf = {}, {}
    print(f"\n  {'regime':<14}{'train rows':>12}{'fraud %':>10}{'AUC-PR':>10}")
    print("  " + "-" * 46)
    for name in ("natural", "undersample", "smote"):
        if name == "natural":
            Xr, yr = Xtr, ytr
        elif name == "undersample":
            Xr, yr = RandomUnderSampler(random_state=SEED).fit_resample(Xtr, ytr)
        else:
            Xr, yr = SMOTE(random_state=SEED, k_neighbors=5).fit_resample(Xtr, ytr)
        m = fit_xgb(Xr, yr)
        ap = average_precision_score(yte, m.predict_proba(Xte)[:, 1])
        models[name], perf[name] = m, ap
        print(f"  {name:<14}{len(yr):>12,}{yr.mean()*100:>9.1f}%{ap:>10.4f}")
        if name != "natural":
            del Xr, yr
            gc.collect()

    ref = models["natural"].predict_proba(Xte)[:, 1]
    idx = np.argsort(-ref)[:N_EXPLAIN]
    sv = {}
    for name, m in models.items():
        v = np.asarray(shap.TreeExplainer(m).shap_values(Xte[idx]))
        sv[name] = v[:, :, 1] if v.ndim == 3 else v

    print(f"\n  --- agreement with NATURAL-imbalance explanations "
          f"({N_EXPLAIN} alerts) ---")
    print(f"  {'regime':<14}{'Spearman':>22}{'Jaccard@' + str(TOP_K):>22}")
    comp, tests = {}, {}
    for name in ("undersample", "smote"):
        rhos = np.array([rank_agreement(sv["natural"][i], sv[name][i])
                         for i in range(N_EXPLAIN)])
        rhos = rhos[np.isfinite(rhos)]
        jac = bootstrap_ci([jaccard_at_k(sv["natural"][i], sv[name][i], TOP_K)
                            for i in range(N_EXPLAIN)])
        rho = bootstrap_ci(rhos)
        print(f"  {name:<14}{rho[0]:>9.3f} [{rho[1]:.3f}, {rho[2]:.3f}]"
              f"{jac[0]:>11.3f} [{jac[1]:.3f}, {jac[2]:.3f}]")
        comp[name] = {"spearman": rho, "jaccard": jac}
        stat, p = wilcoxon(rhos - 1.0, alternative="less")
        d = float((1.0 - rhos.mean()) / (rhos.std(ddof=1) or 1e-9))
        tests[name] = {"p_value": float(p), "mean_rho": float(rhos.mean()),
                       "cohens_d": d, "n": int(len(rhos))}

    print("\n  --- test of H0: rankings invariant to regime ---")
    for name, t in tests.items():
        print(f"  natural vs {name:<12} p = {t['p_value']:.3g}   "
              f"mean rho {t['mean_rho']:.3f}   Cohen's d {t['cohens_d']:.2f}")

    print("\n  --- most-named feature per regime ---")
    named = {}
    for name, v in sv.items():
        top1 = [cols[int(np.argmax(np.abs(v[i])))] for i in range(N_EXPLAIN)]
        vals, counts = np.unique(top1, return_counts=True)
        order = np.argsort(-counts)[:3]
        print(f"  {name:<14}" + ", ".join(
            f"{vals[o]} ({counts[o]*100/N_EXPLAIN:.0f}%)" for o in order))
        named[name] = {str(vals[o]): int(counts[o]) for o in order}

    rejected = all(t["p_value"] < 0.05 for t in tests.values())
    spread = max(perf.values()) - min(perf.values())
    print("\n" + "=" * 78)
    print(f"  H0 {'REJECTED' if rejected else 'not rejected'} on IEEE-CIS   |   "
          f"AUC-PR spread across regimes: {spread:.4f}")
    print("=" * 78)

    OUT.mkdir(exist_ok=True)
    (OUT / "s3_imbalance_explanation_ieee.json").write_text(
        json.dumps({"seed": SEED, "rows_used": N_ROWS, "rows_available": 590540,
                    "subsample_note": "chronological, memory constrained",
                    "performance": perf, "agreement": comp, "tests": tests,
                    "top_features": named, "h0_rejected": rejected},
                   indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote results/s3_imbalance_explanation_ieee.json "
          f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
