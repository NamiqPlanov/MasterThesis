"""
Stage S1 on IEEE-CIS (Secondary Objective 1, second dataset).

Why this dataset matters for the thesis: unlike ULB, whose features are
anonymised principal components, IEEE-CIS has semantically named fields --
TransactionAmt, ProductCD, card4, P_emaildomain, DeviceType. An explanation
over these features can in principle be acted on by an investigator, which is
what Objective 4 sets out to test.

MEMORY NOTE: the full file is 590,540 x 394. On this machine (2.5 GB free) a
float64 load plus the copies made during splitting and scaling does not fit, so
a CHRONOLOGICAL PREFIX is used and columns are read as float32. The subsample
is time-ordered, not random, so the train/test separation still reflects
deployment. This is a documented deviation, recorded in the results file.

    .venv\\Scripts\\python.exe experiments\\05_baselines_ieee.py
"""

from __future__ import annotations

import gc
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
CSV = ROOT / "data" / "ieee" / "train_transaction.csv"
OUT, ART = ROOT / "results", ROOT / "artifacts"
SEED = 42
N_ROWS = 300_000          # chronological prefix, for memory
TEST_FRACTION = 0.30
MAX_MISSING = 0.90        # drop a column emptier than this
ALERT_BUDGETS = (0.001, 0.005, 0.01)


def recall_at_budget(y_true, scores, budget):
    k = max(1, int(len(scores) * budget))
    top = np.argsort(-scores)[:k]
    caught = int(y_true[top].sum())
    total = int(y_true.sum())
    return (caught / total if total else float("nan")), caught, k


def main() -> int:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    ART.mkdir(exist_ok=True)

    print("=" * 78)
    print("  S1 - BASELINE MODELS ON IEEE-CIS (named features)")
    print("=" * 78)
    df = pd.read_csv(CSV, nrows=N_ROWS)
    print(f"  loaded {len(df):,} rows x {df.shape[1]} columns "
          f"(chronological prefix of 590,540)")

    df = df.sort_values("TransactionDT").reset_index(drop=True)
    y = df["isFraud"].to_numpy().astype(np.int8)
    df = df.drop(columns=["isFraud", "TransactionID"])

    # non-numeric fields become integer codes; the NAMES are what matter here
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for c in cat_cols:
        df[c] = pd.factorize(df[c])[0].astype(np.float32)
    print(f"  categorical columns encoded: {len(cat_cols)}  e.g. {cat_cols[:6]}")

    keep = df.columns[df.isna().mean() <= MAX_MISSING]
    dropped = df.shape[1] - len(keep)
    df = df[keep].astype(np.float32)
    print(f"  dropped {dropped} columns over {MAX_MISSING:.0%} missing; "
          f"{df.shape[1]} remain")

    cut = int(len(df) * (1 - TEST_FRACTION))
    cols = list(df.columns)
    X = df.to_numpy(dtype=np.float32)
    del df
    gc.collect()

    med = np.nanmedian(X[:cut], axis=0)
    med = np.where(np.isnan(med), 0.0, med).astype(np.float32)
    X_imp = np.where(np.isnan(X), med, X).astype(np.float32)

    Xtr_raw, Xte_raw = X[:cut], X[cut:]            # NaN kept, for XGBoost
    Xtr_imp, Xte_imp = X_imp[:cut], X_imp[cut:]    # imputed, for LR and RF
    ytr, yte = y[:cut], y[cut:]
    del X_imp
    gc.collect()

    scaler = StandardScaler().fit(Xtr_imp)
    Xtr_s = scaler.transform(Xtr_imp).astype(np.float32)
    Xte_s = scaler.transform(Xte_imp).astype(np.float32)

    print(f"  train / test  : {len(ytr):,} / {len(yte):,}  (chronological)")
    print(f"  fraud in train: {ytr.sum():,} ({ytr.mean() * 100:.3f}%)")
    print(f"  fraud in test : {yte.sum():,} ({yte.mean() * 100:.3f}%)")
    print(f"  always-'legit' accuracy on test: {(1 - yte.mean()) * 100:.3f}%")

    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    specs = [
        ("LogisticRegression", "scaled",
         LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        ("RandomForest", "imputed",
         RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED)),
        ("XGBoost", "raw",
         XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                       subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
                       scale_pos_weight=spw, n_jobs=-1, random_state=SEED,
                       tree_method="hist")),
    ]

    report = {}
    print(f"\n  {'model':<20}{'AUC-PR':>9}{'ROC-AUC':>9}"
          + "".join(f"{'R@' + str(b * 100) + '%':>10}" for b in ALERT_BUDGETS)
          + f"{'fit s':>8}")
    print("  " + "-" * 76)
    for name, kind, model in specs:
        Xa, Xb = {"scaled": (Xtr_s, Xte_s), "imputed": (Xtr_imp, Xte_imp),
                  "raw": (Xtr_raw, Xte_raw)}[kind]
        t = time.time()
        model.fit(Xa, ytr)
        fit_s = time.time() - t
        scores = model.predict_proba(Xb)[:, 1]
        ap = average_precision_score(yte, scores)
        roc = roc_auc_score(yte, scores)
        rec = {}
        for b in ALERT_BUDGETS:
            r, caught, k = recall_at_budget(yte, scores, b)
            rec[f"recall_at_{b}"] = {"recall": r, "caught": caught, "alerts": k,
                                     "of_frauds": int(yte.sum())}
        print(f"  {name:<20}{ap:>9.4f}{roc:>9.4f}"
              + "".join(f"{rec[f'recall_at_{b}']['recall']:>10.3f}" for b in ALERT_BUDGETS)
              + f"{fit_s:>8.1f}")
        report[name] = {"auc_pr": ap, "roc_auc": roc, "fit_seconds": fit_s,
                        "input": kind, **rec}
        joblib.dump(model, ART / f"ieee_{name}.joblib")

    joblib.dump({"columns": cols, "cut": cut, "median": med,
                 "Xte_raw": Xte_raw, "Xte_scaled": Xte_s, "yte": yte,
                 "Xtr_scaled_mean": Xtr_s.mean(axis=0),
                 "Xtr_imp_mean": Xtr_imp.mean(axis=0)},
                ART / "ieee_split.joblib", compress=3)

    best = max(report, key=lambda m: report[m]["auc_pr"])
    print(f"\n  best by AUC-PR: {best} ({report[best]['auc_pr']:.4f})")
    (OUT / "s1_baselines_ieee.json").write_text(
        json.dumps({"seed": SEED, "rows_used": int(N_ROWS),
                    "rows_available": 590540,
                    "subsample_note": "chronological prefix, memory constrained",
                    "features_kept": len(cols), "results": report},
                   indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote results/s1_baselines_ieee.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
