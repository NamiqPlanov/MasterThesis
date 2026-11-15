"""
Why is explanation stability more sensitive to resampling than to reseeding?

S3 showed resampling moves SHAP rankings to rho 0.45-0.58 while S5 showed
reseeding moves them only to 0.873. That is a difference in magnitude without
a mechanism, which is what the supervisor asked for next.

Design: a factorial over {regime} x {seed} x {model configuration}, with every
cell explaining the SAME fixed alert queue. Agreement is then compared between
pairs of cells that differ in exactly ONE factor, which isolates each factor's
contribution without an ANOVA's distributional assumptions.

Four mechanisms are tested:
  M1  the decision function itself changes     -> score-level agreement
  M2  the model's structure changes            -> split-feature usage overlap
  M3  SHAP's reference point moves             -> explainer expected_value
  M4  synthetic points sit off the manifold    -> nearest-neighbour distance

M1 is the discriminating test. If scores barely move while explanations do,
the explanations are not tracking the model's behaviour -- a fidelity problem.
If scores move too, resampling is simply building a different model and the
explanations are right to disagree.

    .venv\\Scripts\\python.exe -u experiments\\16_stability_mechanism.py
"""

from __future__ import annotations

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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from xai_metrics import bootstrap_ci, jaccard_at_k, rank_agreement  # noqa: E402

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

CSV = ROOT / "data" / "ulb" / "creditcard.csv"
OUT = ROOT / "results"
TEST_FRACTION = 0.30
N_ALERT = 400
TOP_K = 5

REGIMES = ("natural", "undersample", "smote")
SEEDS = (42, 7, 13)
# three configurations, so "is this consistent across model configurations?"
# is answered rather than assumed
CONFIGS = {
    "shallow": dict(n_estimators=200, max_depth=3, learning_rate=0.1),
    "default": dict(n_estimators=400, max_depth=6, learning_rate=0.1),
    "deep":    dict(n_estimators=400, max_depth=10, learning_rate=0.05),
}


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


def split_usage(model, n_features):
    """Normalised gain per feature -- how the model's structure uses inputs."""
    gain = model.get_booster().get_score(importance_type="gain")
    v = np.zeros(n_features)
    for key, val in gain.items():
        v[int(key[1:])] = val          # keys look like "f12"
    total = v.sum()
    return v / total if total else v


def main() -> int:
    t0 = time.time()
    df = pd.read_csv(CSV).sort_values("Time").reset_index(drop=True)
    y = df["Class"].to_numpy()
    X = df.drop(columns=["Class"])
    cols = list(X.columns)
    cut = int(len(df) * (1 - TEST_FRACTION))
    scaler = StandardScaler().fit(X.iloc[:cut])
    Xtr, Xte = scaler.transform(X.iloc[:cut]), scaler.transform(X.iloc[cut:])
    ytr, yte = y[:cut], y[cut:]

    print("=" * 78)
    print("  WHY IS RESAMPLING WORSE THAN RESEEDING FOR EXPLANATIONS?")
    print("=" * 78)
    print(f"  cells: {len(REGIMES)} regimes x {len(SEEDS)} seeds x "
          f"{len(CONFIGS)} configs = "
          f"{len(REGIMES) * len(SEEDS) * len(CONFIGS)}", flush=True)

    # the alert queue is fixed once, by the reference cell, so every cell
    # explains the same transactions
    ref = fit_xgb(*resample(Xtr, ytr, "natural", SEEDS[0]),
                  SEEDS[0], CONFIGS["default"])
    idx = np.argsort(-ref.predict_proba(Xte)[:, 1])[:N_ALERT]

    cells = {}
    for regime, seed, cname in itertools.product(REGIMES, SEEDS, CONFIGS):
        t = time.time()
        Xr, yr = resample(Xtr, ytr, regime, seed)
        m = fit_xgb(Xr, yr, seed, CONFIGS[cname])
        s = m.predict_proba(Xte)[:, 1]
        sv = np.asarray(shap.TreeExplainer(m).shap_values(Xte[idx]))
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        ex = shap.TreeExplainer(m)
        cells[(regime, seed, cname)] = {
            "shap": sv,
            "scores": s,
            "auc_pr": float(average_precision_score(yte, s)),
            "expected_value": float(np.ravel(ex.expected_value)[0]),
            "usage": split_usage(m, len(cols)),
            "train_rows": int(len(yr)),
            "train_fraud_rate": float(yr.mean()),
        }
        print(f"    {regime:<12}{seed:<5}{cname:<9}"
              f"AUC-PR {cells[(regime, seed, cname)]['auc_pr']:.4f}  "
              f"({time.time() - t:.0f}s)", flush=True)

    def compare(a, b):
        """Explanation, score and structure agreement between two cells."""
        A, B = cells[a], cells[b]
        rhos = np.array([rank_agreement(A["shap"][i], B["shap"][i])
                         for i in range(N_ALERT)])
        rhos = rhos[np.isfinite(rhos)]
        jac = [jaccard_at_k(A["shap"][i], B["shap"][i], TOP_K)
               for i in range(N_ALERT)]
        changed = [TOP_K - len(set(np.argsort(-np.abs(A["shap"][i]))[:TOP_K])
                               & set(np.argsort(-np.abs(B["shap"][i]))[:TOP_K]))
                   for i in range(N_ALERT)]
        return {
            "shap_spearman": bootstrap_ci(rhos),
            "shap_jaccard": bootstrap_ci(jac),
            # the practically meaningful number: of the five features an
            # investigator reads, how many are different?
            "top5_features_changed": bootstrap_ci(changed),
            "score_spearman": float(spearmanr(A["scores"], B["scores"]).statistic),
            "score_spearman_alerts": float(
                spearmanr(A["scores"][idx], B["scores"][idx]).statistic),
            "usage_cosine": float(
                A["usage"] @ B["usage"]
                / (np.linalg.norm(A["usage"]) * np.linalg.norm(B["usage"]) + 1e-12)),
            "expected_value_delta": abs(A["expected_value"] - B["expected_value"]),
            "auc_pr_delta": abs(A["auc_pr"] - B["auc_pr"]),
        }

    # pairs differing in exactly one factor
    groups = {"seed_only": [], "regime_only": [], "config_only": []}
    for a, b in itertools.combinations(cells, 2):
        diff = [a[i] != b[i] for i in range(3)]
        if sum(diff) != 1:
            continue
        key = ("regime_only", "seed_only", "config_only")[diff.index(True)]
        groups[key].append(compare(a, b))

    print("\n  --- agreement when exactly one factor changes ---")
    print(f"  {'factor':<14}{'n':>4}{'SHAP rho':>11}{'top5 changed':>15}"
          f"{'score rho':>12}{'usage cos':>11}{'dAUC-PR':>10}")
    summary = {}
    for key, rows in groups.items():
        if not rows:
            continue
        summary[key] = {
            "n_pairs": len(rows),
            "shap_spearman_mean": float(np.mean([r["shap_spearman"][0] for r in rows])),
            "top5_changed_mean": float(np.mean([r["top5_features_changed"][0] for r in rows])),
            "score_spearman_mean": float(np.mean([r["score_spearman"] for r in rows])),
            "usage_cosine_mean": float(np.mean([r["usage_cosine"] for r in rows])),
            "auc_pr_delta_mean": float(np.mean([r["auc_pr_delta"] for r in rows])),
            "expected_value_delta_mean": float(np.mean([r["expected_value_delta"] for r in rows])),
        }
        s = summary[key]
        print(f"  {key:<14}{s['n_pairs']:>4}{s['shap_spearman_mean']:>11.3f}"
              f"{s['top5_changed_mean']:>15.2f}{s['score_spearman_mean']:>12.4f}"
              f"{s['usage_cosine_mean']:>11.3f}{s['auc_pr_delta_mean']:>10.4f}")

    # M4: do SMOTE's synthetic points sit off the data manifold?
    nn = NearestNeighbors(n_neighbors=1).fit(Xtr[ytr == 1])
    Xs, ys = resample(Xtr, ytr, "smote", SEEDS[0])
    synth = Xs[len(Xtr):]
    manifold = {}
    if len(synth):
        d_syn = nn.kneighbors(synth[ys[len(Xtr):] == 1][:5000])[0].ravel()
        d_real = nn.kneighbors(Xtr[ytr == 1], n_neighbors=2)[0][:, 1]
        manifold = {"median_nn_distance_synthetic": float(np.median(d_syn)),
                    "median_nn_distance_real_fraud": float(np.median(d_real)),
                    "ratio": float(np.median(d_syn) / (np.median(d_real) + 1e-12))}
        print(f"\n  SMOTE synthetic points sit {manifold['ratio']:.2f}x the "
              "typical fraud-to-fraud distance from real fraud")

    (OUT / "s6_stability_mechanism.json").write_text(
        json.dumps({"regimes": list(REGIMES), "seeds": list(SEEDS),
                    "configs": {k: v for k, v in CONFIGS.items()},
                    "n_alert": N_ALERT, "top_k": TOP_K,
                    "one_factor_summary": summary,
                    "smote_manifold": manifold,
                    "cells": {f"{r}|{s}|{c}": {
                        "auc_pr": cells[(r, s, c)]["auc_pr"],
                        "expected_value": cells[(r, s, c)]["expected_value"],
                        "train_rows": cells[(r, s, c)]["train_rows"],
                        "train_fraud_rate": cells[(r, s, c)]["train_fraud_rate"]}
                        for r, s, c in cells}},
                   indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote results/s6_stability_mechanism.json "
          f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())