"""
Random-sample deletion fidelity on IEEE-CIS -- closing the asymmetry in S2.

03_* reports fidelity on BOTH populations on ULB (alert queue and random
sample); 06_* reports the alert queue only. That asymmetry is a residual
inconsistency in experimental conditions of exactly the kind the supervisor's
review asked to eliminate, and the random-sample figure is the one that shows
how misleading a fidelity number quoted over ordinary transactions is.

This lives in its own script rather than being folded into 06_* because 06_*
costs roughly five hours of CPU, almost all of it in the 1,000-instance
max-sensitivity loop over a 390-feature forest, whereas the measurement below
takes minutes. 06_* has been amended so that a future run computes it inline;
this script supplies the number for the run already completed, and writes to
its own results file so nothing overwrites a measurement it did not make.

The instance ordering is reproduced exactly from 06_*: the same seed, the same
XGBoost-ranked alert queue on the raw matrix, the same 600 random draws from
the complement. The random sample is therefore the identical set of rows.

    .venv\\Scripts\\python.exe -u experiments\\14_fidelity_random_ieee.py
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from xai_metrics import bootstrap_ci, deletion_fidelity  # noqa: E402

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

ART, OUT = ROOT / "artifacts", ROOT / "results"
SEED = 42
N_ALERT, N_RANDOM = 400, 600
CURVE_K = 20            # identical to 03_* and 06_*


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    split = joblib.load(ART / "ieee_split.joblib")
    cols, yte = split["columns"], split["yte"]
    Xte_raw, Xte_s = split["Xte_raw"], split["Xte_scaled"]

    print("=" * 78)
    print("  RANDOM-SAMPLE DELETION FIDELITY ON IEEE-CIS")
    print("=" * 78)
    print(f"  features: {len(cols)}  |  random instances: {N_RANDOM}  "
          f"|  curve_k: {CURVE_K}", flush=True)

    models = {n: joblib.load(ART / f"ieee_{n}.joblib")
              for n in ("LogisticRegression", "RandomForest", "XGBoost")}

    med = split["median"]
    Xte_imp = np.where(np.isnan(Xte_raw), med, Xte_raw).astype(np.float32)
    inputs = {"LogisticRegression": Xte_s, "RandomForest": Xte_imp,
              "XGBoost": Xte_raw}
    baselines = {"LogisticRegression": split["Xtr_scaled_mean"],
                 "RandomForest": split["Xtr_imp_mean"],
                 "XGBoost": np.nan_to_num(split["Xtr_imp_mean"])}

    # reproduce 06_*'s ordering exactly so the random sample is the same rows
    ref = models["XGBoost"].predict_proba(Xte_raw)[:, 1]
    alert_idx = np.argsort(-ref)[:N_ALERT]
    pool = np.setdiff1d(np.arange(len(yte)), alert_idx)
    rand_idx = rng.choice(pool, N_RANDOM, replace=False)
    print(f"  frauds in the random sample: {int(yte[rand_idx].sum())} "
          f"of {N_RANDOM}\n", flush=True)

    report = {}
    for name, model in models.items():
        t = time.time()
        Xm, base = inputs[name], baselines[name]
        if name == "LogisticRegression":
            ex = shap.LinearExplainer(
                model, shap.maskers.Independent(Xm, max_samples=2000))
        else:
            ex = shap.TreeExplainer(model)
        sv = np.asarray(ex.shap_values(Xm[rand_idx]))
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        def predict(a, m=model):
            return m.predict_proba(a)[:, 1]

        fr = bootstrap_ci([
            deletion_fidelity(predict, Xm[rand_idx[i]], sv[i], base, k=CURVE_K)
            for i in range(N_RANDOM)])
        scores = model.predict_proba(Xm[rand_idx])[:, 1]
        print(f"  {name:<22}{fr[0]:>8.3f}  [{fr[1]:.3f}, {fr[2]:.3f}]   "
              f"median score {np.median(scores):.2e}   "
              f"({time.time() - t:.0f}s)", flush=True)
        report[name] = {"fidelity_random": fr,
                        "median_predicted_score": float(np.median(scores))}

    (OUT / "s2_fidelity_random_ieee.json").write_text(
        json.dumps({"seed": SEED, "n_random": N_RANDOM, "curve_k": CURVE_K,
                    "n_features": len(cols), "per_model": report},
                   indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote results/s2_fidelity_random_ieee.json "
          f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
