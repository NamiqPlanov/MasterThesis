"""
Stage S5: does the explanation survive retraining with a different random seed?

The supervisor asked for explanation stability "across multiple runs". This is a
DIFFERENT question from the max-sensitivity measured in S2, and the distinction
matters:

  S2 max-sensitivity   perturbs the INPUT, model fixed  -> is the explanation
                       robust to a slightly different transaction?
  S5 (this script)     perturbs the SEED, input fixed   -> is the explanation
                       robust to a different run of the same training recipe?

Both are stability. They fail for different reasons, and a method can pass one
while failing the other.

The split is rebuilt from the CSV exactly as 02_baselines_ulb.py builds it --
chronological sort on Time, the same 70/30 cut, scaler fitted on train only --
because the artifacts store the test side only. Reusing the identical split is
what makes this comparable to the baselines rather than confounded with them.

H0: SHAP rankings are invariant to the training seed.

    .venv\\Scripts\\python.exe experiments\\12_stability_seeds.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from xai_metrics import bootstrap_ci, jaccard_at_k, rank_agreement  # noqa: E402

CSV = ROOT / "data" / "ulb" / "creditcard.csv"
OUT = ROOT / "results"

SEEDS = [42, 7, 13, 21, 34, 55, 89, 101, 202, 303]
TEST_FRACTION = 0.30
N_ALERT = 400
TOP_K = 5


def fit_xgb(X, y, seed):
    """Identical to the spec in 02_baselines_ulb.py apart from random_state."""
    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
        scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)),
        n_jobs=-1, random_state=seed, tree_method="hist").fit(X, y)


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print("  S5 - IS THE EXPLANATION STABLE ACROSS RETRAINING SEEDS?")
    print("=" * 78)

    df = pd.read_csv(CSV).sort_values("Time").reset_index(drop=True)
    y = df["Class"].to_numpy()
    X = df.drop(columns=["Class"])
    cols = list(X.columns)

    cut = int(len(df) * (1 - TEST_FRACTION))
    Xtr_raw, Xte_raw = X.iloc[:cut], X.iloc[cut:]
    ytr, yte = y[:cut], y[cut:]

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)

    print(f"  train / test : {len(ytr):,} / {len(yte):,}  (chronological)")
    print(f"  fraud in test: {yte.sum():,} ({yte.mean() * 100:.3f}%)")
    print(f"  seeds        : {len(SEEDS)}   alerts explained: {N_ALERT}")

    sv, perf, idx = {}, {}, None
    print(f"\n  {'seed':<8}{'AUC-PR':>10}{'fit s':>9}")
    print("  " + "-" * 27)
    for s in SEEDS:
        t = time.time()
        m = fit_xgb(Xtr, ytr, s)
        scores = m.predict_proba(Xte)[:, 1]
        perf[s] = float(average_precision_score(yte, scores))
        if idx is None:
            # the alert queue is fixed by the reference seed, so every seed is
            # explaining the SAME transactions
            idx = np.argsort(-scores)[:N_ALERT]
        v = np.asarray(shap.TreeExplainer(m).shap_values(Xte[idx]))
        sv[s] = v[:, :, 1] if v.ndim == 3 else v
        print(f"  {s:<8}{perf[s]:>10.4f}{time.time() - t:>9.1f}")

    ref_seed = SEEDS[0]
    ref = sv[ref_seed]

    print(f"\n  --- agreement with seed {ref_seed} ({N_ALERT} alerts) ---")
    print(f"  {'seed':<8}{'Spearman':>24}{'Jaccard@' + str(TOP_K):>24}")
    agree, tests = {}, {}
    for s in SEEDS[1:]:
        rhos = np.array([rank_agreement(ref[i], sv[s][i])
                         for i in range(N_ALERT)])
        rhos = rhos[np.isfinite(rhos)]
        rho = bootstrap_ci(rhos)
        jac = bootstrap_ci([jaccard_at_k(ref[i], sv[s][i], TOP_K)
                            for i in range(N_ALERT)])
        agree[str(s)] = {"spearman": rho, "jaccard": jac}
        stat, p = wilcoxon(rhos - 1.0, alternative="less")
        tests[str(s)] = {"p_value": float(p), "mean_rho": float(rhos.mean()),
                         "cohens_d": float((1.0 - rhos.mean())
                                           / (rhos.std(ddof=1) or 1e-9)),
                         "n": int(len(rhos))}
        print(f"  {s:<8}{rho[0]:>10.3f} [{rho[1]:.3f}, {rho[2]:.3f}]"
              f"{jac[0]:>12.3f} [{jac[1]:.3f}, {jac[2]:.3f}]")

    print("\n  --- most-named feature per seed ---")
    named = {}
    for s in SEEDS:
        top1 = [cols[int(np.argmax(np.abs(sv[s][i])))] for i in range(N_ALERT)]
        vals, counts = np.unique(top1, return_counts=True)
        order = np.argsort(-counts)[:3]
        named[str(s)] = {str(vals[o]): int(counts[o]) for o in order}
        print(f"  {s:<8}" + ", ".join(
            f"{vals[o]} ({counts[o] * 100 / N_ALERT:.0f}%)" for o in order))

    mean_rho = float(np.mean([t["mean_rho"] for t in tests.values()]))
    rejected = all(t["p_value"] < 0.05 for t in tests.values())
    spread = max(perf.values()) - min(perf.values())

    print("\n" + "=" * 78)
    print(f"  mean Spearman across seeds : {mean_rho:.3f}")
    print(f"  AUC-PR spread across seeds : {spread:.4f} "
          f"(min {min(perf.values()):.4f}, max {max(perf.values()):.4f})")
    print(f"  H0 (rankings invariant to seed): "
          f"{'REJECTED' if rejected else 'NOT rejected'}")
    print("=" * 78)

    OUT.mkdir(exist_ok=True)
    (OUT / "s5_stability_seeds.json").write_text(
        json.dumps({"seeds": SEEDS, "reference_seed": ref_seed,
                    "n_alert": N_ALERT, "top_k": TOP_K,
                    "test_fraction": TEST_FRACTION,
                    "performance": {str(k): v for k, v in perf.items()},
                    "auc_pr_spread": spread,
                    "agreement": agree, "tests": tests,
                    "mean_spearman": mean_rho,
                    "top_features": named,
                    "h0_rejected": rejected},
                   indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote results/s5_stability_seeds.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
