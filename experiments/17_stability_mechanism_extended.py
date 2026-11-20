"""
Does the resampling > reseeding asymmetry hold across datasets and explainers?

16_* established the ordering on ULB with SHAP, across three model
configurations. The supervisor asked whether it is consistent across
explanation methods and datasets too, which 16_* cannot answer: it is
ULB-only and SHAP-only.

This runs the same one-factor design on BOTH datasets and computes BOTH
explainers in every cell, giving four independent (dataset, explainer)
replications of the same comparison. If the ordering survives all four, the
finding generalises; if it holds for SHAP but not LIME, the effect is partly
a property of the explainer rather than of the model, which would matter more.

Cells are 3 regimes x 2 seeds x 2 configurations = 12 per dataset, reduced
from 16_*'s 27 because the cost now doubles twice over. Expect roughly an
hour in total; IEEE-CIS with LIME dominates it.

    .venv\\Scripts\\python.exe -u experiments\\17_stability_mechanism_extended.py
"""

from __future__ import annotations

import gc
import itertools
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from xai_metrics import (  # noqa: E402
    bootstrap_ci, jaccard_at_k, rank_agreement, topk_changed,
)
from lime_fit import best_fit_attribution, build_explainers  # noqa: E402

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

OUT = ROOT / "results"
ULB_CSV = ROOT / "data" / "ulb" / "creditcard.csv"
IEEE_CSV = ROOT / "data" / "ieee" / "train_transaction.csv"

TEST_FRACTION = 0.30
IEEE_ROWS = 150_000
MAX_MISSING = 0.90
N_ALERT = 400           # instances explained with SHAP
N_LIME = 100            # LIME is far slower; a subsample with intervals
LIME_SAMPLES = 800
TOP_K = 5
MIN_R2 = 0.5            # drop LIME attributions whose surrogate does not fit

REGIMES = ("natural", "undersample", "smote")
SEEDS = (42, 7)
CONFIGS = {"default": dict(n_estimators=300, max_depth=6, learning_rate=0.1),
           "deep": dict(n_estimators=300, max_depth=10, learning_rate=0.05)}


def fit_xgb(X, y, seed, cfg):
    pos = max(int((y == 1).sum()), 1)
    return XGBClassifier(
        subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
        scale_pos_weight=float((y == 0).sum() / pos),
        n_jobs=-1, random_state=seed, tree_method="hist", **cfg).fit(X, y)


def resample(Xtr, ytr, regime, seed):
    if regime == "natural":
        return Xtr, ytr
    if regime == "undersample":
        return RandomUnderSampler(random_state=seed).fit_resample(Xtr, ytr)
    return SMOTE(random_state=seed, k_neighbors=5).fit_resample(Xtr, ytr)


def load_ulb():
    """Chronological 70/30 split, scaler on train only -- as 02_* builds it."""
    df = pd.read_csv(ULB_CSV).sort_values("Time").reset_index(drop=True)
    y = df["Class"].to_numpy()
    X = df.drop(columns=["Class"])
    cols = list(X.columns)
    cut = int(len(df) * (1 - TEST_FRACTION))
    scaler = StandardScaler().fit(X.iloc[:cut])
    return (scaler.transform(X.iloc[:cut]).astype(np.float32),
            scaler.transform(X.iloc[cut:]).astype(np.float32),
            y[:cut], y[cut:], cols)


def load_ieee():
    """Identical preprocessing to 10_*: factorise, drop >90% empty, impute."""
    df = pd.read_csv(IEEE_CSV, nrows=IEEE_ROWS).sort_values("TransactionDT")
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
    out = X[:cut], X[cut:], y[:cut], y[cut:], cols
    del X
    gc.collect()
    return out


def run_dataset(label, loader):
    t0 = time.time()
    Xtr, Xte, ytr, yte, cols = loader()
    print("=" * 78)
    print(f"  {label}: train {len(ytr):,} / test {len(yte):,}, "
          f"{len(cols)} features, {yte.sum():,} test frauds")
    print(f"  cells: {len(REGIMES)}x{len(SEEDS)}x{len(CONFIGS)} = "
          f"{len(REGIMES) * len(SEEDS) * len(CONFIGS)}", flush=True)

    # the alert queue is fixed once by the reference cell, so every cell in
    # this dataset explains the same transactions
    ref = fit_xgb(*resample(Xtr, ytr, "natural", SEEDS[0]),
                  SEEDS[0], CONFIGS["default"])
    idx = np.argsort(-ref.predict_proba(Xte)[:, 1])[:N_ALERT]
    del ref
    gc.collect()

    # one LIME explainer per candidate kernel width, reused across cells
    lime_exps = build_explainers(Xte, cols, SEEDS[0])

    cells = {}
    for regime, seed, cname in itertools.product(REGIMES, SEEDS, CONFIGS):
        t = time.time()
        Xr, yr = resample(Xtr, ytr, regime, seed)
        m = fit_xgb(Xr, yr, seed, CONFIGS[cname])
        del Xr, yr
        gc.collect()

        scores = m.predict_proba(Xte)[:, 1]
        sv = np.asarray(shap.TreeExplainer(m).shap_values(Xte[idx]))
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        # LIME on the head of the same queue, width chosen per instance
        lv, lr2 = [], []
        for i in range(N_LIME):
            v, r2, _mult, _by = best_fit_attribution(
                lime_exps, Xte[idx[i]], m.predict_proba, LIME_SAMPLES)
            lv.append(v)
            lr2.append(r2)

        cells[(regime, seed, cname)] = {
            "shap": sv, "lime": np.asarray(lv), "lime_r2": np.asarray(lr2),
            "scores": scores,
            "auc_pr": float(average_precision_score(yte, scores)),
        }
        print(f"    {regime:<12}{seed:<5}{cname:<9}"
              f"AUC-PR {cells[(regime, seed, cname)]['auc_pr']:.4f}  "
              f"LIME R2 med {np.median(lr2):.3f}  "
              f"({time.time() - t:.0f}s)", flush=True)
        del m
        gc.collect()

    def compare(a, b, explainer):
        A, B = cells[a], cells[b]
        n = N_ALERT if explainer == "shap" else N_LIME
        va, vb = A[explainer], B[explainer]
        # for LIME, only instances whose surrogate fits in BOTH cells
        keep = (range(n) if explainer == "shap" else
                [i for i in range(n)
                 if A["lime_r2"][i] >= MIN_R2 and B["lime_r2"][i] >= MIN_R2])
        keep = list(keep)
        if not keep:
            return None
        rhos = np.array([rank_agreement(va[i], vb[i]) for i in keep])
        rhos = rhos[np.isfinite(rhos)]
        if not len(rhos):
            return None
        return {
            "spearman": bootstrap_ci(rhos),
            "jaccard": bootstrap_ci([jaccard_at_k(va[i], vb[i], TOP_K)
                                     for i in keep]),
            "top5_changed": bootstrap_ci([topk_changed(va[i], vb[i], TOP_K)
                                          for i in keep]),
            "score_spearman": float(
                spearmanr(A["scores"], B["scores"]).statistic),
            "auc_pr_delta": abs(A["auc_pr"] - B["auc_pr"]),
            "n_used": len(keep),
        }

    summary = {}
    for explainer in ("shap", "lime"):
        groups = {"seed_only": [], "regime_only": [], "config_only": []}
        for a, b in itertools.combinations(cells, 2):
            diff = [a[i] != b[i] for i in range(3)]
            if sum(diff) != 1:
                continue
            res = compare(a, b, explainer)
            if res:
                key = ("regime_only", "seed_only", "config_only")[diff.index(True)]
                groups[key].append(res)

        print(f"\n  --- {label} / {explainer.upper()}: "
              "one factor changed at a time ---")
        print(f"  {'factor':<16}{'pairs':>6}{'rho':>9}{'top5 chg':>11}"
              f"{'score rho':>11}{'dAUC-PR':>10}")
        summary[explainer] = {}
        for key, rows in groups.items():
            if not rows:
                continue
            s = {
                "n_pairs": len(rows),
                "spearman_mean": float(np.mean([r["spearman"][0] for r in rows])),
                "jaccard_mean": float(np.mean([r["jaccard"][0] for r in rows])),
                "top5_changed_mean": float(
                    np.mean([r["top5_changed"][0] for r in rows])),
                "score_spearman_mean": float(
                    np.mean([r["score_spearman"] for r in rows])),
                "auc_pr_delta_mean": float(
                    np.mean([r["auc_pr_delta"] for r in rows])),
                "n_used_mean": float(np.mean([r["n_used"] for r in rows])),
            }
            summary[explainer][key] = s
            print(f"  {key:<16}{s['n_pairs']:>6}{s['spearman_mean']:>9.3f}"
                  f"{s['top5_changed_mean']:>11.2f}"
                  f"{s['score_spearman_mean']:>11.3f}"
                  f"{s['auc_pr_delta_mean']:>10.4f}")

        # the question the supervisor actually asked
        r = summary[explainer].get("regime_only", {}).get("spearman_mean")
        sd = summary[explainer].get("seed_only", {}).get("spearman_mean")
        if r is not None and sd is not None:
            holds = r < sd
            print(f"  ordering (regime moves more than seed): "
                  f"{'HOLDS' if holds else 'DOES NOT HOLD'}  "
                  f"({r:.3f} vs {sd:.3f})")
            summary[explainer]["ordering_holds"] = bool(holds)

    print(f"\n  {label} done in {time.time() - t0:.0f}s\n", flush=True)
    return {"n_features": len(cols), "n_train": int(len(ytr)),
            "n_test": int(len(yte)), "summary": summary}


def main() -> int:
    report = {"regimes": list(REGIMES), "seeds": list(SEEDS),
              "configs": CONFIGS, "n_alert": N_ALERT, "n_lime": N_LIME,
              "lime_samples": LIME_SAMPLES, "min_local_r2": MIN_R2,
              "top_k": TOP_K, "datasets": {}}
    for label, loader in (("ULB", load_ulb), ("IEEE-CIS", load_ieee)):
        report["datasets"][label] = run_dataset(label, loader)

    (OUT / "s6_stability_mechanism_extended.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")
    print("  wrote results/s6_stability_mechanism_extended.json")

    print("\n  === the consistency answer ===")
    for label, ds in report["datasets"].items():
        for ex, s in ds["summary"].items():
            if "ordering_holds" in s:
                print(f"  {label:<10} {ex.upper():<5} "
                      f"{'holds' if s['ordering_holds'] else 'DOES NOT HOLD'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())