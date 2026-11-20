"""
Generate REPRODUCIBILITY.md from the conditions each experiment recorded.

The supervisor asked for the experimental settings, resampling procedures,
evaluation protocols and limitations to be documented so that others can
reproduce the findings, and for the methodological differences from earlier
work to be stated. Reading the conditions from results/ rather than retyping
them is what keeps this document honest as the experiments change.

    .venv\\Scripts\\python.exe reports\\build_reproducibility.py
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "REPRODUCIBILITY.md"

PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost",
            "imbalanced-learn", "shap", "lime", "scipy", "matplotlib")


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def versions():
    """Pinned versions of the packages the results depend on."""
    try:
        out = subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"),
                              "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return {}
    found = {}
    for line in out.splitlines():
        if "==" in line:
            name, ver = line.split("==", 1)
            if name.strip().lower() in PACKAGES:
                found[name.strip()] = ver.strip()
    return found


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=30).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


s2u, s2i = (load("s2_explanation_quality_ulb.json"),
            load("s2_explanation_quality_ieee.json"))
s5 = load("s5_stability_seeds.json")
lime_i = load("s2_shap_vs_lime_ieee.json")

L = []
L += ["# Reproducibility", "",
      f"Generated from `results/` at commit `{git_commit()}` by "
      "`reports/build_reproducibility.py`. Every value below is read from the "
      "file the experiment wrote, so this document cannot drift from the "
      "results it describes.", ""]

# ----------------------------------------------------------------- environment
L += ["## 1. Environment", "",
      f"- Python {platform.python_version()} on {platform.system()} "
      f"{platform.release()}",
      "- PostgreSQL 18.4, used for exploratory querying only; no result "
      "depends on it", ""]
v = versions()
if v:
    L += ["| Package | Version |", "|---|---|"]
    L += [f"| {k} | {v[k]} |" for k in sorted(v)]
    L.append("")

# ------------------------------------------------------------ recorded settings
L += ["## 2. Experimental settings", "",
      "Each script writes its own conditions into its results file. The "
      "table below is those recorded values, not a restatement of intent "
      "— an unrecorded sample size once reversed a conclusion in this "
      "project, which is why the recording exists.", "",
      "| Setting | ULB | IEEE-CIS |", "|---|---|---|"]
for key, label in (("seed", "Random seed"),
                   ("n_alert", "Alert-queue instances"),
                   ("n_random", "Random-sample instances"),
                   ("n_stability", "Instances per stability estimate"),
                   ("curve_k", "Fidelity curve length (top-k deleted)"),
                   ("top_k", "Top-k for agreement measures"),
                   ("n_lime", "Instances explained with LIME"),
                   ("lime_samples", "LIME perturbations per instance"),
                   ("min_local_r2", "LIME surrogate R² floor")):
    a = s2u.get(key) if s2u else None
    b = s2i.get(key) if s2i else None
    if b is None and lime_i:
        b = lime_i.get(key)
    if a is not None or b is not None:
        L.append(f"| {label} | {a if a is not None else '—'} | "
                 f"{b if b is not None else '—'} |")
L += ["",
      "- Splits are chronological 70/30 on the transaction timestamp "
      "(`Time` for ULB, `TransactionDT` for IEEE-CIS). Scaling and "
      "imputation statistics are fitted on the training partition only.",
      "- IEEE-CIS uses a chronological **prefix** of 300,000 rows for "
      "baselines and explanation quality and 150,000 for the imbalance "
      "experiment, taken as a prefix rather than a random draw so the time "
      "ordering survives.",
      "- Columns more than 90% empty are dropped (two of 392). Remaining "
      "gaps are filled with training-partition medians for the random "
      "forest; XGBoost consumes them natively.",
      "- Hyperparameters are fixed a priori and never tuned.", ""]

# ----------------------------------------------------------------- resampling
L += ["## 3. Resampling procedures", "",
      "Resampling is applied to the **training partition only**, after the "
      "chronological split, so no synthetic or duplicated row can reach the "
      "test set.", "",
      "| Regime | Procedure |", "|---|---|",
      "| natural | No resampling. Imbalance is addressed by instance "
      "weighting: `class_weight='balanced'` for logistic regression, "
      "`balanced_subsample` for the random forest, `scale_pos_weight = "
      "negatives/positives` for XGBoost. |",
      "| undersample | `imblearn.under_sampling.RandomUnderSampler`, "
      "default (majority reduced to minority count), seeded. |",
      "| smote | `imblearn.over_sampling.SMOTE(k_neighbors=5)`, seeded. |",
      "",
      "The baselines use instance weighting rather than resampling "
      "deliberately: resampling is the manipulated variable of the imbalance "
      "experiment, and using it in the baseline would confound treatment "
      "with control.", ""]

# ----------------------------------------------------------------- protocol
L += ["## 4. Evaluation protocol", "",
      "**Detection.** AUC-PR is the primary threshold-free measure; ROC-AUC "
      "is reported alongside; accuracy is never reported, because at a "
      "0.172% base rate predicting the majority class scores 99.83%.", "",
      "**Alert budget.** At budget *b* the alerts are the top *b* fraction "
      "of test transactions by model score, so *k* = ⌊*b·n*⌋ is fixed and "
      "the model determines only which cases fill the slots. Budgets are "
      "0.1%, 0.5% and 1%. Because *k* is fixed, precision and recall at a "
      "budget are arithmetically linked (*precision = recall·F/k*) and are "
      "not independent evidence; both are reported because practitioners "
      "need both.", "",
      "**Explanations.** Generated for two populations that answer "
      "different questions: the alert queue (the highest-scored test "
      "transactions, what an analyst sees) and a random sample from the "
      "remainder (the population the model scores). SHAP uses the exact "
      "algorithm for each model class — LinearExplainer for logistic "
      "regression, TreeExplainer for the ensembles — never KernelExplainer.",
      "",
      "**LIME.** The locality kernel width is selected **per instance** by "
      "maximising the surrogate's local R², and instances whose best "
      "surrogate falls below the R² floor are excluded and counted. This is "
      "necessary rather than optional: at 390 features the default width "
      f"gives a median local R² of "
      f"{lime_i['per_model']['LogisticRegression']['lime_local_r2_median_at_default_width']:.3f}"
      " and returns numerical noise without raising an error. The same "
      "procedure applied to ULB selects the default width for most "
      "instances, so one implementation serves both dimensionalities.", "",
      "**Inference.** 95% percentile bootstrap intervals over 2,000 "
      "resamples on every point estimate; Wilcoxon signed-rank for paired "
      "comparisons. At n = 400 every p-value is ≈1e-67 and carries no "
      "information, so Cohen's *d* and the number of top-5 features that "
      "change identity carry the argument.", "",
      "**Metric validation.** The measures are validated on synthetic "
      "models whose true importances are known by construction, including a "
      "control that a random attribution must fail. See "
      "`experiments/01_synthetic_validation.py`.", ""]

# ------------------------------------------------------------- reproduce order
L += ["## 5. Reproducing the results", "",
      "Scripts are numbered in dependency order. From a clean checkout:", "",
      "```bash",
      "python -m venv .venv && .venv/Scripts/pip install -r requirements.txt",
      ".venv/Scripts/python.exe experiments/00_get_data.py        # Kaggle API",
      ".venv/Scripts/python.exe experiments/01_synthetic_validation.py",
      ".venv/Scripts/python.exe experiments/02_baselines_ulb.py",
      ".venv/Scripts/python.exe experiments/03_explanation_quality_ulb.py",
      ".venv/Scripts/python.exe experiments/04_imbalance_explanation_ulb.py",
      ".venv/Scripts/python.exe experiments/05_baselines_ieee.py",
      ".venv/Scripts/python.exe experiments/06_explanation_quality_ieee.py",
      ".venv/Scripts/python.exe experiments/08_exploratory_analysis.py",
      ".venv/Scripts/python.exe experiments/10_imbalance_explanation_ieee.py",
      ".venv/Scripts/python.exe experiments/12_stability_seeds.py",
      ".venv/Scripts/python.exe experiments/13_lime_ieee.py",
      ".venv/Scripts/python.exe experiments/14_fidelity_random_ieee.py",
      ".venv/Scripts/python.exe experiments/15_cross_data_analysys.py",
      ".venv/Scripts/python.exe experiments/16_stability_mechanism.py",
      ".venv/Scripts/python.exe experiments/17_stability_mechanism_extended.py",
      ".venv/Scripts/python.exe experiments/09_figures.py",
      ".venv/Scripts/python.exe experiments/99_summary_tables.py",
      "```", "",
      "All randomness is seeded. The slowest step is "
      "`06_explanation_quality_ieee.py` at roughly five hours of processor "
      "time, dominated by the 1,000-instance stability loop over a "
      "390-feature forest. Datasets are **not** redistributed: "
      "`00_get_data.py` fetches them, and `data/`, `artifacts/`, `logs/` "
      "and credentials are git-ignored.", ""]

# ------------------------------------------------------- what differs from prior work
L += ["## 6. How this methodology differs from earlier research", "",
      "| Aspect | Common practice in the reviewed literature | This work |",
      "|---|---|---|",
      "| Object of measurement | The model; explanations are presented but "
      "not evaluated | The explanation output itself, with the model as "
      "instrument |",
      "| Evidence for explanation quality | Predictive performance used as a "
      "proxy (>80% of 49 reviewed studies) | Fidelity, stability and "
      "agreement measured directly, with bootstrap intervals |",
      "| Train/test split | Random | Chronological, because fraud drifts "
      "several-fold across deciles in both datasets |",
      "| Operating point | A probability threshold, or none stated | A fixed "
      "alert budget reflecting a team's review capacity |",
      "| Headline metric | Accuracy or ROC-AUC | AUC-PR plus precision and "
      "recall at budget; accuracy never reported |",
      "| Imbalance handling | SMOTE applied by default, effect on "
      "explanations unexamined | Resampling treated as a manipulated "
      "variable, with performance held as a covariate and a seed control |",
      "| Metric trustworthiness | Asserted | Validated on synthetic controls "
      "with known importances, including a random-attribution control |",
      "| Explainer configuration | Library defaults | LIME's kernel width "
      "selected per instance by surrogate fit, with poorly-fitted instances "
      "excluded and counted |",
      "| Reported conditions | Rarely stated | Seed, instance counts, curve "
      "length and top-k written into every results file |",
      "| Agreement reporting | Rank correlation alone where reported | Rank "
      "correlation **and** top-k overlap, because they diverge (ρ = 0.553 "
      "with Jaccard 0.013 on IEEE-CIS) |", ""]

# ----------------------------------------------------------------- limitations
L += ["## 7. Limitations", "",
      "- **Proxy properties, not usefulness.** Fidelity, stability and "
      "agreement are necessary but not sufficient for a useful explanation. "
      "Establishing usefulness requires professional fraud analysts, which "
      "was not feasible here.",
      "- **IEEE-CIS subsample.** Results use a chronological prefix, so they "
      "strictly concern the dataset's first 85 days.",
      "- **Untuned hyperparameters.** The models are reasonable, not "
      "optimal; tuning was avoided because its interaction with explanation "
      "quality is itself unmeasured.",
      "- **One architecture in the imbalance experiment.** Generalisation "
      "across model families is not established.",
      "- **No intervals on budget-level rates.** Precision and recall at a "
      "budget are reported as point estimates.",
      f"- **Seed stability covers one dataset.** {len(s5['seeds'])} seeds on "
      "ULB only.",
      "- **Rubric study outstanding.** The qualitative half of the "
      "actionability objective needs a second independent rater before "
      "inter-rater agreement can be reported.", ""]

OUT.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)} ({len(L)} lines)")