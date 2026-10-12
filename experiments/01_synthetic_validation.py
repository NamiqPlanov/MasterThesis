"""
Stage S4 control: validate the explanation-quality metrics on models whose
true feature importances are known by construction.

Proposal, section 4.1: "a metric that cannot recover a known answer cannot be
trusted on an unknown one". Before any of these measures are applied to a fraud
model, they must demonstrably recover importances we planted ourselves.

The synthetic data deliberately mimics fraud: ~2% positive rate, 20 features of
which only 5 carry signal.

    .venv\\Scripts\\python.exe experiments\\01_synthetic_validation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from xai_metrics import (  # noqa: E402
    bootstrap_ci, faithfulness, ground_truth_recovery,
    jaccard_at_k, max_sensitivity, rank_agreement,
)

SEED = 42
N_SAMPLES = 6000
N_FEATURES = 20
TRUE_COEFS = np.array([3.0, -2.0, 1.5, -1.0, 0.5] + [0.0] * 15)
N_EXPLAIN = 120          # instances to explain per model
TARGET_POS_RATE = 0.02   # fraud-like imbalance


def make_data(rng):
    X = rng.normal(size=(N_SAMPLES, N_FEATURES))
    logit = X @ TRUE_COEFS
    # shift the intercept until the positive rate matches the target
    intercept = np.quantile(logit, 1 - TARGET_POS_RATE)
    p = 1 / (1 + np.exp(-(logit - intercept)))
    y = (rng.uniform(size=N_SAMPLES) < p).astype(int)
    return X, y


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    X, y = make_data(rng)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    baseline = Xtr.mean(axis=0)

    print("=" * 74)
    print("  S4 CONTROL - can the metrics recover importances we planted?")
    print("=" * 74)
    print(f"  {N_SAMPLES:,} rows x {N_FEATURES} features | "
          f"positives {y.mean() * 100:.2f}% | informative features: 5")
    print(f"  true coefficients: {TRUE_COEFS[:5]}")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
    }

    report = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        ap = average_precision_score(yte, model.predict_proba(Xte)[:, 1])

        if name == "LogisticRegression":
            # the default masker subsamples the background to 100 rows, which
            # shifts E[x] and breaks an exact comparison against the closed
            # form; force it to use the whole training set
            masker = shap.maskers.Independent(Xtr, max_samples=len(Xtr))
            explainer = shap.LinearExplainer(model, masker)
        else:
            explainer = shap.TreeExplainer(model)

        idx = rng.choice(len(Xte), N_EXPLAIN, replace=False)
        sample = Xte[idx]
        sv = explainer.shap_values(sample)
        sv = np.asarray(sv)
        if sv.ndim == 3:                     # (n, features, classes)
            sv = sv[:, :, 1]

        def predict(a, m=model):
            return m.predict_proba(a)[:, 1]

        def explain_one(x, e=explainer):
            v = np.asarray(e.shap_values(x.reshape(1, -1)))
            if v.ndim == 3:
                v = v[:, :, 1]
            return v.reshape(-1)

        # SHAP explains the MODEL, not the data-generating process. Two
        # different questions, and conflating them invalidates the control:
        #   (a) does SHAP reproduce what this model actually computes?
        #   (b) does the model's global importance match the planted signal?
        global_imp = np.abs(sv).mean(axis=0)
        # Rank correlation over all 20 features is depressed by construction:
        # the 15 noise features are tied at a true importance of 0, so no
        # ordering of them can be "correct". The meaningful test is whether the
        # 5 features that do carry signal are ranked in the planted order.
        n_inf = int((TRUE_COEFS != 0).sum())
        global_recovery = rank_agreement(global_imp[:n_inf], np.abs(TRUE_COEFS[:n_inf]))
        global_recovery_all = rank_agreement(global_imp, np.abs(TRUE_COEFS))
        global_jaccard = jaccard_at_k(global_imp, np.abs(TRUE_COEFS), 5)

        exactness = None
        if name == "LogisticRegression":
            # for a linear model the closed form is coef * (x - E[x]);
            # LinearExplainer must reproduce it to numerical precision
            expected = model.coef_[0] * (sample - Xtr.mean(axis=0))
            exactness = float(np.max(np.abs(sv - expected)))

        recov = [ground_truth_recovery(row, np.abs(TRUE_COEFS)) for row in sv]
        faith = [faithfulness(predict, sample[i], sv[i], baseline)
                 for i in range(len(sample))]
        stab = [max_sensitivity(explain_one, sample[i], eps=0.05, n_samples=10,
                                rng=np.random.default_rng(i))
                for i in range(25)]          # costly: a 25-instance subsample

        sp = bootstrap_ci([r["spearman_vs_truth"] for r in recov])
        jc = bootstrap_ci([r["jaccard_at_k"] for r in recov])
        nm = bootstrap_ci([r["noise_mass"] for r in recov])
        fa = bootstrap_ci(faith)
        st = bootstrap_ci(stab)

        print(f"\n  --- {name} (test AUC-PR {ap:.3f}) ---")
        print(f"  {'informative-5 ranking rho':<34}{global_recovery:>8.3f}")
        print(f"  {'all-20 ranking rho (ties inflate)':<34}{global_recovery_all:>8.3f}")
        print(f"  {'global Jaccard@5 vs planted':<34}{global_jaccard:>8.3f}")
        if exactness is not None:
            print(f"  {'max |SHAP - closed form|':<34}{exactness:>8.2e}")
        print(f"  {'metric':<34}{'mean':>8}{'95% CI':>22}")
        for label, (m, lo, hi) in (
                ("Spearman vs true importance", sp),
                ("Jaccard@5 vs true top-5", jc),
                ("mass on the 15 noise features", nm),
                ("faithfulness (ins-del AUC)", fa),
                ("max-sensitivity (lower better)", st)):
            print(f"  {label:<34}{m:>8.3f}      [{lo:.3f}, {hi:.3f}]")

        report[name] = {
            "auc_pr": ap,
            "global_recovery_rho": global_recovery,
            "global_recovery_rho_all": global_recovery_all,
            "global_jaccard_at_5": global_jaccard,
            "closed_form_max_error": exactness,
            "per_instance_spearman_vs_planted": sp,
            "jaccard_at_5": jc, "noise_mass": nm,
            "faithfulness": fa, "max_sensitivity": st,
        }

    # ---- control: a random attribution must fail these metrics -------------
    model = models["LogisticRegression"]
    idx = rng.choice(len(Xte), 100, replace=False)
    rand_attr = rng.normal(size=(100, N_FEATURES))
    rand_recov = [ground_truth_recovery(r, np.abs(TRUE_COEFS)) for r in rand_attr]
    rand_faith = [faithfulness(lambda a: model.predict_proba(a)[:, 1],
                               Xte[i], rand_attr[j], baseline)
                  for j, i in enumerate(idx)]
    rs = bootstrap_ci([r["spearman_vs_truth"] for r in rand_recov])
    rf = bootstrap_ci(rand_faith)
    print("\n  --- NEGATIVE CONTROL: random attributions ---")
    print(f"  {'Spearman vs true importance':<34}{rs[0]:>8.3f}      [{rs[1]:.3f}, {rs[2]:.3f}]")
    print(f"  {'faithfulness (ins-del AUC)':<34}{rf[0]:>8.3f}      [{rf[1]:.3f}, {rf[2]:.3f}]")
    report["RandomAttribution"] = {"spearman_vs_truth": rs, "faithfulness": rf}

    # ---- cross-explainer agreement on the same model -----------------------
    lin_sv = np.asarray(shap.LinearExplainer(
        models["LogisticRegression"], Xtr).shap_values(Xte[:100]))
    rf_sv = np.asarray(shap.TreeExplainer(
        models["RandomForest"]).shap_values(Xte[:100]))
    if rf_sv.ndim == 3:
        rf_sv = rf_sv[:, :, 1]
    agree = bootstrap_ci([rank_agreement(lin_sv[i], rf_sv[i]) for i in range(100)])
    jac = bootstrap_ci([jaccard_at_k(lin_sv[i], rf_sv[i], 5) for i in range(100)])
    print("\n  --- agreement between the two models' explanations ---")
    print(f"  {'Spearman (LR vs RF)':<34}{agree[0]:>8.3f}      [{agree[1]:.3f}, {agree[2]:.3f}]")
    print(f"  {'Jaccard@5 (LR vs RF)':<34}{jac[0]:>8.3f}      [{jac[1]:.3f}, {jac[2]:.3f}]")
    report["cross_model_agreement"] = {"spearman": agree, "jaccard_at_5": jac}

    # ---- verdict -----------------------------------------------------------
    lr, rf = report["LogisticRegression"], report["RandomForest"]
    checks = {
        "LinearExplainer matches closed form (<1e-6)":
            lr["closed_form_max_error"] < 1e-6,
        "LR ranks the 5 informative features correctly (rho > 0.9)":
            lr["global_recovery_rho"] > 0.90,
        "RF ranks the 5 informative features correctly (rho > 0.8)":
            rf["global_recovery_rho"] > 0.80,
        "LR global top-5 = planted top-5 (Jaccard = 1.0)":
            lr["global_jaccard_at_5"] >= 0.99,
        "SHAP beats random on faithfulness":
            lr["faithfulness"][0] > report["RandomAttribution"]["faithfulness"][0],
        "random attribution fails recovery (|rho| < 0.3)":
            abs(report["RandomAttribution"]["spearman_vs_truth"][0]) < 0.30,
        "stability finite and ordered (LR smoother than RF)":
            lr["max_sensitivity"][0] < rf["max_sensitivity"][0],
    }
    print("\n" + "=" * 74)
    print("  VERDICT")
    print("=" * 74)
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    passed = all(checks.values())
    print(f"\n  {'metrics validated - safe to apply to real data'if passed else 'IMPLEMENTATION SUSPECT - do not use on real data yet'}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "s4_synthetic_validation.json").write_text(
        json.dumps({"seed": SEED, "checks": checks, "report": report},
                   indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote results/s4_synthetic_validation.json   ({time.time() - t0:.0f}s)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
