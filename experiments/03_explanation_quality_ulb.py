"""
Stage S2 (Secondary Objective 2): measure the explanations, not the model.

This is the step Progress Report 2 found missing from the literature: over 80%
of studies report model accuracy and then present SHAP plots without asking
whether those plots are faithful, stable, or consistent with one another.

Measured on the models fitted in S1, on the same time-ordered test split:
  fidelity   -- insertion minus deletion AUC
  stability  -- max-sensitivity in an eps-ball
  agreement  -- SHAP vs LIME on one model, and model vs model

Instances are drawn from two populations, because they answer different
questions: the ALERT QUEUE (what an analyst actually sees) and a RANDOM
sample (the population the model scores).

    .venv\\Scripts\\python.exe experiments\\03_explanation_quality_ulb.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from xai_metrics import (  # noqa: E402
    bootstrap_ci, faithfulness, jaccard_at_k, max_sensitivity, rank_agreement,
)

warnings.filterwarnings("ignore")
ART, OUT = ROOT / "artifacts", ROOT / "results"
SEED = 42
N_ALERT, N_RANDOM = 400, 600      # ~1,000 instances, as the proposal specifies
N_LIME = 120                      # LIME is far slower; subsample with CIs
N_STABILITY = 80
TOP_K = 5
CURVE_K = 20            # fidelity curve length; identical across datasets


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    split = joblib.load(ART / "ulb_split.joblib")
    Xte, yte, cols = split["Xte"], split["yte"], split["columns"]
    baseline = split["Xtr_mean"]

    models = {n: joblib.load(ART / f"ulb_{n}.joblib")
              for n in ("LogisticRegression", "RandomForest", "XGBoost")}

    print("=" * 78)
    print("  S2 - EXPLANATION QUALITY ON ULB")
    print("=" * 78)

    # the alert queue = what a 0.5%-budget analyst would review, per best model
    ref_scores = models["RandomForest"].predict_proba(Xte)[:, 1]
    alert_idx = np.argsort(-ref_scores)[:N_ALERT]
    pool = np.setdiff1d(np.arange(len(Xte)), alert_idx)
    rand_idx = rng.choice(pool, N_RANDOM, replace=False)
    idx_all = np.concatenate([alert_idx, rand_idx])
    print(f"  instances explained : {len(idx_all)} "
          f"({N_ALERT} alert queue + {N_RANDOM} random)")
    print(f"  frauds in alert set : {int(yte[alert_idx].sum())} of {int(yte.sum())}")

    lime_exp = LimeTabularExplainer(
        training_data=Xte, feature_names=cols, class_names=["legit", "fraud"],
        discretize_continuous=False, random_state=SEED, mode="classification")

    shap_vals, report = {}, {}
    for name, model in models.items():
        t = time.time()
        if name == "LogisticRegression":
            ex = shap.LinearExplainer(
                model, shap.maskers.Independent(Xte, max_samples=len(Xte)))
        else:
            ex = shap.TreeExplainer(model)
        sv = np.asarray(ex.shap_values(Xte[idx_all]))
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        shap_vals[name] = sv

        def predict(a, m=model):
            return m.predict_proba(a)[:, 1]

        faith_alert = [faithfulness(predict, Xte[idx_all[i]], sv[i], baseline, k=CURVE_K)
                       for i in range(N_ALERT)]
        faith_rand = [faithfulness(predict, Xte[idx_all[i]], sv[i], baseline, k=CURVE_K)
                      for i in range(N_ALERT, len(idx_all))]

        def explain_batch(a, e=ex):
            v = np.asarray(e.shap_values(a))
            if v.ndim == 3:
                v = v[:, :, 1]
            return v

        stab = [max_sensitivity(explain_batch, Xte[idx_all[i]], eps=0.05,
                                n_samples=8, rng=np.random.default_rng(i))
                for i in range(N_STABILITY)]

        # SHAP vs LIME on the SAME model and the same instances
        agree_rho, agree_jac = [], []
        for i in range(N_LIME):
            j = idx_all[i]
            e = lime_exp.explain_instance(
                Xte[j], model.predict_proba, num_features=len(cols), num_samples=800)
            lv = np.zeros(len(cols))
            for fidx, w in e.as_map()[1]:
                lv[fidx] = w
            agree_rho.append(rank_agreement(sv[i], lv))
            agree_jac.append(jaccard_at_k(sv[i], lv, TOP_K))

        fa_a, fa_r = bootstrap_ci(faith_alert), bootstrap_ci(faith_rand)
        st = bootstrap_ci(stab)
        ar, aj = bootstrap_ci(agree_rho), bootstrap_ci(agree_jac)

        print(f"\n  --- {name}  ({time.time() - t:.0f}s) ---")
        print(f"  {'measure':<38}{'mean':>8}{'95% CI':>20}")
        for lbl, v in (("fidelity, alert queue", fa_a),
                       ("fidelity, random sample", fa_r),
                       ("max-sensitivity (lower better)", st),
                       ("SHAP vs LIME, Spearman", ar),
                       (f"SHAP vs LIME, Jaccard@{TOP_K}", aj)):
            print(f"  {lbl:<38}{v[0]:>8.3f}     [{v[1]:.3f}, {v[2]:.3f}]")

        report[name] = {"fidelity_alert": fa_a, "fidelity_random": fa_r,
                        "max_sensitivity": st,
                        "shap_vs_lime_spearman": ar,
                        "shap_vs_lime_jaccard": aj}

    # ---- do different models tell the same story about the same alert? -----
    print("\n  --- cross-model agreement (alert queue only) ---")
    pairs = [("LogisticRegression", "RandomForest"),
             ("LogisticRegression", "XGBoost"),
             ("RandomForest", "XGBoost")]
    cross = {}
    for a, b in pairs:
        rho = bootstrap_ci([rank_agreement(shap_vals[a][i], shap_vals[b][i])
                            for i in range(N_ALERT)])
        jac = bootstrap_ci([jaccard_at_k(shap_vals[a][i], shap_vals[b][i], TOP_K)
                            for i in range(N_ALERT)])
        print(f"  {a[:12]:>12} vs {b[:12]:<14} rho {rho[0]:>6.3f}  "
              f"Jaccard@{TOP_K} {jac[0]:>5.3f}")
        cross[f"{a}|{b}"] = {"spearman": rho, "jaccard": jac}

    # ---- which features does each model name most often? -------------------
    print("\n  --- most-named feature in the alert queue ---")
    named = {}
    for name, sv in shap_vals.items():
        top1 = [cols[int(np.argmax(np.abs(sv[i])))] for i in range(N_ALERT)]
        vals, counts = np.unique(top1, return_counts=True)
        order = np.argsort(-counts)[:3]
        s = ", ".join(f"{vals[o]} ({counts[o] * 100 / N_ALERT:.0f}%)" for o in order)
        print(f"  {name:<20}{s}")
        named[name] = {str(vals[o]): int(counts[o]) for o in order}

    (OUT / "s2_explanation_quality_ulb.json").write_text(
        json.dumps({"seed": SEED, "n_alert": N_ALERT, "n_random": N_RANDOM,
                    "n_lime": N_LIME, "per_model": report,
                    "cross_model": cross, "top_features": named},
                   indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote results/s2_explanation_quality_ulb.json  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
