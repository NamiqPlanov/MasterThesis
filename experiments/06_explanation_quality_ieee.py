"""
Stage S2 on IEEE-CIS, and the evidence for Objective 4.

Same measurements as 03_*, but on a dataset whose features have names. The
comparison is the point of the thesis: on ULB an explanation says "V14", which
no investigator can act on; here it can say "P_emaildomain" or "TransactionAmt".

Objective 4 asks whether that difference is real. This script produces the
quantitative half -- how often each model names a SEMANTIC feature rather than
an anonymised engineered one -- and writes the alert explanations that the
two human raters will score.

    .venv\\Scripts\\python.exe experiments\\06_explanation_quality_ieee.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from xai_metrics import (  # noqa: E402
    bootstrap_ci, deletion_fidelity, jaccard_at_k, max_sensitivity, rank_agreement,
)

warnings.filterwarnings("ignore")
ART, OUT = ROOT / "artifacts", ROOT / "results"
SEED = 42
N_ALERT, N_RANDOM = 400, 600
N_STABILITY = 80
TOP_K = 5
CURVE_K = 20            # fidelity curve length; identical across datasets

# IEEE-CIS feature families. V1-V339 are Vesta's engineered features and id_*
# are masked: both are opaque in exactly the way ULB's V1-V28 are. Everything
# else carries a meaning an investigator could act on.
OPAQUE = re.compile(r"^(V\d+|id_\d+|C\d+|D\d+)$")


def is_semantic(name: str) -> bool:
    return not OPAQUE.match(name)


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    split = joblib.load(ART / "ieee_split.joblib")
    cols = split["columns"]
    yte = split["yte"]
    Xte_raw, Xte_s = split["Xte_raw"], split["Xte_scaled"]

    n_sem = sum(is_semantic(c) for c in cols)
    print("=" * 78)
    print("  S2 / OBJECTIVE 4 - EXPLANATION QUALITY ON IEEE-CIS")
    print("=" * 78)
    print(f"  features: {len(cols)}  |  semantic: {n_sem}  "
          f"|  opaque (V*/id_*/C*/D*): {len(cols) - n_sem}")

    models = {n: joblib.load(ART / f"ieee_{n}.joblib")
              for n in ("LogisticRegression", "RandomForest", "XGBoost")}
    inputs = {"LogisticRegression": Xte_s, "RandomForest": Xte_raw,
              "XGBoost": Xte_raw}
    # RandomForest was fitted on imputed (not scaled) data; rebuild that view
    med = split["median"]
    Xte_imp = np.where(np.isnan(Xte_raw), med, Xte_raw).astype(np.float32)
    inputs["RandomForest"] = Xte_imp
    baselines = {"LogisticRegression": split["Xtr_scaled_mean"],
                 "RandomForest": split["Xtr_imp_mean"],
                 "XGBoost": np.nan_to_num(split["Xtr_imp_mean"])}

    ref = models["XGBoost"].predict_proba(Xte_raw)[:, 1]
    alert_idx = np.argsort(-ref)[:N_ALERT]
    pool = np.setdiff1d(np.arange(len(yte)), alert_idx)
    rand_idx = rng.choice(pool, N_RANDOM, replace=False)
    idx_all = np.concatenate([alert_idx, rand_idx])
    print(f"  explained: {len(idx_all)} instances "
          f"({N_ALERT} alert queue, {int(yte[alert_idx].sum())} of them fraud)")

    shap_vals, report, semantic_share = {}, {}, {}
    for name, model in models.items():
        t = time.time()
        Xm, base = inputs[name], baselines[name]
        if name == "LogisticRegression":
            ex = shap.LinearExplainer(
                model, shap.maskers.Independent(Xm, max_samples=2000))
        else:
            ex = shap.TreeExplainer(model)
        sv = np.asarray(ex.shap_values(Xm[idx_all]))
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        shap_vals[name] = sv

        def predict(a, m=model):
            return m.predict_proba(a)[:, 1]

        fa = bootstrap_ci([deletion_fidelity(predict, Xm[idx_all[i]], sv[i], base, k=CURVE_K)
                           for i in range(N_ALERT)])

        def explain_batch(a, e=ex):
            v = np.asarray(e.shap_values(a))
            if v.ndim == 3:
                v = v[:, :, 1]
            return v

        st = bootstrap_ci([max_sensitivity(explain_batch, Xm[idx_all[i]], eps=0.05,
                                           n_samples=8,
                                           rng=np.random.default_rng(i))
                           for i in range(N_STABILITY)])

        # the Objective 4 number: is the top-ranked feature one a human can use?
        tops = [cols[int(np.argmax(np.abs(sv[i])))] for i in range(N_ALERT)]
        share = float(np.mean([is_semantic(c) for c in tops]))
        top5_share = float(np.mean([
            np.mean([is_semantic(cols[j])
                     for j in np.argsort(-np.abs(sv[i]))[:TOP_K]])
            for i in range(N_ALERT)]))
        semantic_share[name] = {"top1": share, "top5": top5_share}

        vals, counts = np.unique(tops, return_counts=True)
        order = np.argsort(-counts)[:4]
        print(f"\n  --- {name}  ({time.time() - t:.0f}s) ---")
        print(f"  {'fidelity (alert queue)':<34}{fa[0]:>8.3f}  [{fa[1]:.3f}, {fa[2]:.3f}]")
        print(f"  {'max-sensitivity':<34}{st[0]:>8.3f}  [{st[1]:.3f}, {st[2]:.3f}]")
        print(f"  {'top-1 feature is SEMANTIC':<34}{share * 100:>7.1f}%")
        print(f"  {'top-5 features semantic (mean)':<34}{top5_share * 100:>7.1f}%")
        print("  most-named: " + ", ".join(
            f"{vals[o]} ({counts[o] * 100 / N_ALERT:.0f}%)" for o in order))

        report[name] = {"fidelity_alert": fa, "max_sensitivity": st,
                        "semantic_top1": share, "semantic_top5": top5_share,
                        "most_named": {str(vals[o]): int(counts[o]) for o in order}}

    print("\n  --- cross-model agreement (alert queue) ---")
    cross = {}
    for a, b in (("LogisticRegression", "RandomForest"),
                 ("LogisticRegression", "XGBoost"),
                 ("RandomForest", "XGBoost")):
        rho = bootstrap_ci([rank_agreement(shap_vals[a][i], shap_vals[b][i])
                            for i in range(N_ALERT)])
        jac = bootstrap_ci([jaccard_at_k(shap_vals[a][i], shap_vals[b][i], TOP_K)
                            for i in range(N_ALERT)])
        print(f"  {a[:12]:>12} vs {b[:12]:<14} rho {rho[0]:>6.3f}   "
              f"Jaccard@{TOP_K} {jac[0]:>5.3f}")
        cross[f"{a}|{b}"] = {"spearman": rho, "jaccard": jac}

    (OUT / "s2_explanation_quality_ieee.json").write_text(
        json.dumps({"seed": SEED, "n_features": len(cols),
                    "n_semantic_features": n_sem,
                    "per_model": report, "cross_model": cross,
                    "semantic_share": semantic_share},
                   indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote results/s2_explanation_quality_ieee.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
