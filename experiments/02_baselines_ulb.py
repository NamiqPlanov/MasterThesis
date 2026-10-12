"""
Stage S1 (Secondary Objective 1): reproducible modelling baseline on ULB.

Two choices here differ from most of the literature reviewed in Progress
Report 2, and both are deliberate:

  * the split is TIME-ORDERED, not random. Fraud adapts, so a random split
    lets the model learn from transactions that happen after the ones it is
    tested on -- optimistic and unlike deployment.
  * performance is AUC-PR and recall at a fixed alert budget, not accuracy.
    At a 0.17% base rate a model that always answers "legitimate" scores
    99.83% accurate and catches nothing.

    .venv\\Scripts\\python.exe experiments\\02_baselines_ulb.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "ulb" / "creditcard.csv"
OUT = ROOT / "results"
ART = ROOT / "artifacts"
SEED = 42
TEST_FRACTION = 0.30
ALERT_BUDGETS = (0.001, 0.005, 0.01)     # review capacity as a share of volume


def recall_at_budget(y_true, scores, budget):
    """Of all frauds, what share lands in the top `budget` of ranked alerts?

    This is the number a fraud team actually lives with: analysts can only
    review so many cases a day.
    """
    k = max(1, int(len(scores) * budget))
    top = np.argsort(-scores)[:k]
    caught = int(y_true[top].sum())
    total = int(y_true.sum())
    return caught / total if total else float("nan"), caught, k


def main() -> int:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    ART.mkdir(exist_ok=True)

    df = pd.read_csv(CSV)
    df = df.sort_values("Time").reset_index(drop=True)   # chronological order
    y = df["Class"].to_numpy()
    X = df.drop(columns=["Class"])

    cut = int(len(df) * (1 - TEST_FRACTION))
    Xtr_raw, Xte_raw = X.iloc[:cut], X.iloc[cut:]
    ytr, yte = y[:cut], y[cut:]

    # V1-V28 are already PCA components; Time and Amount are not scaled
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)

    print("=" * 78)
    print("  S1 - BASELINE MODELS ON ULB (time-ordered split)")
    print("=" * 78)
    print(f"  total rows      : {len(df):,}   features: {X.shape[1]}")
    print(f"  train / test    : {len(ytr):,} / {len(yte):,}  (chronological)")
    print(f"  fraud in train  : {ytr.sum():,}  ({ytr.mean() * 100:.3f}%)")
    print(f"  fraud in test   : {yte.sum():,}  ({yte.mean() * 100:.3f}%)")
    print(f"  always-'legit' accuracy on test: {(1 - yte.mean()) * 100:.3f}%"
          "   <- why accuracy is not reported")

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
            scale_pos_weight=float((ytr == 0).sum() / max((ytr == 1).sum(), 1)),
            n_jobs=-1, random_state=SEED, tree_method="hist"),
    }

    report = {}
    print(f"\n  {'model':<20}{'AUC-PR':>9}{'ROC-AUC':>9}"
          + "".join(f"{'R@' + str(b * 100) + '%':>10}" for b in ALERT_BUDGETS)
          + f"{'fit s':>8}")
    print("  " + "-" * 76)
    for name, model in models.items():
        t = time.time()
        model.fit(Xtr, ytr)
        fit_s = time.time() - t
        scores = model.predict_proba(Xte)[:, 1]
        ap = average_precision_score(yte, scores)
        roc = roc_auc_score(yte, scores)
        rec = {}
        for b in ALERT_BUDGETS:
            r, caught, k = recall_at_budget(yte, scores, b)
            rec[f"recall_at_{b}"] = {"recall": r, "caught": caught,
                                     "alerts": k, "of_frauds": int(yte.sum())}
        print(f"  {name:<20}{ap:>9.4f}{roc:>9.4f}"
              + "".join(f"{rec[f'recall_at_{b}']['recall']:>10.3f}" for b in ALERT_BUDGETS)
              + f"{fit_s:>8.1f}")
        report[name] = {"auc_pr": ap, "roc_auc": roc, "fit_seconds": fit_s, **rec}
        joblib.dump(model, ART / f"ulb_{name}.joblib")

    # keep the split so the explanation stage measures the same model/data
    joblib.dump({"scaler": scaler, "Xte": Xte, "yte": yte,
                 "columns": list(X.columns), "cut": cut,
                 "Xtr_mean": Xtr.mean(axis=0)}, ART / "ulb_split.joblib")

    best = max(report, key=lambda m: report[m]["auc_pr"])
    print(f"\n  best by AUC-PR: {best} ({report[best]['auc_pr']:.4f})")
    b = ALERT_BUDGETS[1]
    r = report[best][f"recall_at_{b}"]
    print(f"  at a {b * 100:.1f}% alert budget it reviews {r['alerts']:,} transactions "
          f"and catches {r['caught']} of {r['of_frauds']} frauds ({r['recall'] * 100:.1f}%)")

    (OUT / "s1_baselines_ulb.json").write_text(
        json.dumps({"seed": SEED, "split": "time-ordered",
                    "test_fraction": TEST_FRACTION, "results": report},
                   indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote results/s1_baselines_ulb.json   ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
