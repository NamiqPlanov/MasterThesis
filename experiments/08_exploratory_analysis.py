"""
Stage S1 (October): exploratory analysis of both datasets.

Documents the properties that shape every later decision in the thesis: how
rare fraud is, how that rate moves over time, how much of each dataset is
missing, and how much of it is semantically meaningful. Several of these
numbers are quoted in the proposal and the literature review, so they are
computed here from the data rather than from memory.

    .venv\\Scripts\\python.exe experiments\\08_exploratory_analysis.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
ULB = ROOT / "data" / "ulb" / "creditcard.csv"
IEEE = ROOT / "data" / "ieee" / "train_transaction.csv"
IEEE_ROWS = 300_000          # same prefix the IEEE experiments use
OPAQUE = re.compile(r"^(V\d+|id_\d+|C\d+|D\d+)$")
N_BINS = 10


def drift_table(y, order, n_bins=N_BINS):
    """Fraud rate across equal-sized chronological slices."""
    idx = np.argsort(order)
    y = np.asarray(y)[idx]
    chunks = np.array_split(y, n_bins)
    return [float(c.mean()) for c in chunks]


def main() -> int:
    report = {}
    print("=" * 76)
    print("  S1 - EXPLORATORY ANALYSIS")
    print("=" * 76)

    # ------------------------------------------------------------- ULB -----
    df = pd.read_csv(ULB)
    y = df["Class"].to_numpy()
    amt = df["Amount"].to_numpy()
    hours = df["Time"].to_numpy() / 3600.0
    drift = drift_table(y, df["Time"].to_numpy())

    print("\n  ULB Credit Card Fraud Detection")
    print(f"    rows                 {len(df):,}")
    print(f"    features             {df.shape[1] - 1}  "
          f"(V1-V28 are PCA components; only Time and Amount are raw)")
    print(f"    fraud                {int(y.sum()):,}  ({y.mean() * 100:.3f}%)")
    print(f"    duplicate rows       {int(df.duplicated().sum()):,}")
    print(f"    missing values       {int(df.isna().sum().sum()):,}")
    print(f"    time span            {hours.max():.1f} hours")
    print(f"    amount, legitimate   median {np.median(amt[y == 0]):.2f}   "
          f"mean {amt[y == 0].mean():.2f}   max {amt[y == 0].max():,.0f}")
    print(f"    amount, fraudulent   median {np.median(amt[y == 1]):.2f}   "
          f"mean {amt[y == 1].mean():.2f}   max {amt[y == 1].max():,.0f}")
    print(f"    zero-amount frauds   {int(((amt == 0) & (y == 1)).sum())}")
    print(f"    fraud rate by decile {['%.3f%%' % (d * 100) for d in drift]}")
    print(f"    drift (max/min)      {max(drift) / max(min(drift), 1e-9):.1f}x")

    corr = {c: float(np.corrcoef(df[c], y)[0, 1])
            for c in df.columns if c != "Class"}
    top = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:5]
    print(f"    strongest |corr| with fraud: "
          + ", ".join(f"{k} {v:+.3f}" for k, v in top))

    report["ULB"] = {
        "rows": len(df), "features": int(df.shape[1] - 1),
        "fraud": int(y.sum()), "fraud_rate": float(y.mean()),
        "duplicates": int(df.duplicated().sum()),
        "missing": int(df.isna().sum().sum()),
        "span_hours": float(hours.max()),
        "amount_median_legit": float(np.median(amt[y == 0])),
        "amount_median_fraud": float(np.median(amt[y == 1])),
        "fraud_rate_by_decile": drift,
        "semantic_features": 2, "opaque_features": 28,
        "top_correlations": dict(top),
    }
    del df

    # ----------------------------------------------------------- IEEE ------
    df = pd.read_csv(IEEE, nrows=IEEE_ROWS)
    y = df["isFraud"].to_numpy()
    amt = df["TransactionAmt"].to_numpy()
    drift = drift_table(y, df["TransactionDT"].to_numpy())
    feats = [c for c in df.columns if c not in ("isFraud", "TransactionID")]
    semantic = [c for c in feats if not OPAQUE.match(c)]
    miss = df[feats].isna().mean()

    print("\n  IEEE-CIS Fraud Detection  (chronological prefix)")
    print(f"    rows                 {len(df):,} of 590,540")
    print(f"    features             {len(feats)}   "
          f"semantic {len(semantic)}   opaque {len(feats) - len(semantic)}")
    print(f"    fraud                {int(y.sum()):,}  ({y.mean() * 100:.3f}%)")
    print(f"    duplicate rows       {int(df.duplicated(subset=feats).sum()):,}")
    print(f"    missing overall      {miss.mean() * 100:.1f}% of cells")
    print(f"    columns >50% missing {int((miss > 0.5).sum())}   "
          f">90% missing {int((miss > 0.9).sum())}")
    print(f"    span (days)          "
          f"{(df['TransactionDT'].max() - df['TransactionDT'].min()) / 86400:.1f}")
    print(f"    amount, legitimate   median {np.median(amt[y == 0]):.2f}   "
          f"mean {amt[y == 0].mean():.2f}")
    print(f"    amount, fraudulent   median {np.median(amt[y == 1]):.2f}   "
          f"mean {amt[y == 1].mean():.2f}")
    print(f"    fraud rate by decile {['%.2f%%' % (d * 100) for d in drift]}")
    print(f"    drift (max/min)      {max(drift) / max(min(drift), 1e-9):.1f}x")
    print(f"    semantic fields      {semantic[:10]}")

    report["IEEE"] = {
        "rows_used": len(df), "rows_available": 590540,
        "features": len(feats), "semantic_features": len(semantic),
        "opaque_features": len(feats) - len(semantic),
        "fraud": int(y.sum()), "fraud_rate": float(y.mean()),
        "missing_share": float(miss.mean()),
        "cols_over_50pc_missing": int((miss > 0.5).sum()),
        "amount_median_legit": float(np.median(amt[y == 0])),
        "amount_median_fraud": float(np.median(amt[y == 1])),
        "fraud_rate_by_decile": drift,
        "semantic_field_names": semantic,
    }

    # --------------------------------------------------------- takeaways ---
    print("\n" + "=" * 76)
    print("  WHAT THIS MEANS FOR THE THESIS")
    print("=" * 76)
    u, i = report["ULB"], report["IEEE"]
    print(f"  * Accuracy is useless here: predicting 'legitimate' always scores "
          f"{(1 - u['fraud_rate']) * 100:.2f}% on ULB.")
    print(f"  * Fraud rate is NOT stationary: it varies "
          f"{max(u['fraud_rate_by_decile']) / max(min(u['fraud_rate_by_decile']), 1e-9):.0f}x "
          f"across ULB and "
          f"{max(i['fraud_rate_by_decile']) / max(min(i['fraud_rate_by_decile']), 1e-9):.0f}x "
          f"across IEEE-CIS over time,")
    print("    which a random split would hide. This is why splits are "
          "time-ordered (Gap 5).")
    print(f"  * Interpretability is structural, not a dataset choice: ULB has "
          f"{u['semantic_features']} meaningful")
    print(f"    features of {u['features']}, IEEE-CIS {i['semantic_features']} "
          f"of {i['features']}. Neither can support")
    print("    an explanation an investigator could act on (Gap 4).")
    print(f"  * ULB contains {u['duplicates']:,} duplicate rows, which inflate "
          "results if unnoticed.")

    OUT.mkdir(exist_ok=True)
    (OUT / "s1_exploratory_analysis.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote results/s1_exploratory_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
