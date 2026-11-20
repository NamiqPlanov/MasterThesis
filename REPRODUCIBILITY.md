# Reproducibility

Generated from `results/` at commit `6ccdc6d` by `reports/build_reproducibility.py`. Every value below is read from the file the experiment wrote, so this document cannot drift from the results it describes.

## 1. Environment

- Python 3.13.9 on Windows 11
- PostgreSQL 18.4, used for exploratory querying only; no result depends on it

| Package | Version |
|---|---|
| imbalanced-learn | 0.14.2 |
| lime | 0.2.0.1 |
| matplotlib | 3.11.2 |
| numpy | 2.5.4 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| shap | 0.53.0 |
| xgboost | 3.4.1 |

## 2. Experimental settings

Each script writes its own conditions into its results file. The table below is those recorded values, not a restatement of intent — an unrecorded sample size once reversed a conclusion in this project, which is why the recording exists.

| Setting | ULB | IEEE-CIS |
|---|---|---|
| Random seed | 42 | 42 |
| Alert-queue instances | 400 | 400 |
| Random-sample instances | 600 | 600 |
| Instances per stability estimate | 1000 | 1000 |
| Fidelity curve length (top-k deleted) | 20 | 20 |
| Top-k for agreement measures | 5 | 5 |
| Instances explained with LIME | 120 | 120 |
| LIME perturbations per instance | 800 | 2000 |
| LIME surrogate R² floor | 0.5 | 0.5 |

- Splits are chronological 70/30 on the transaction timestamp (`Time` for ULB, `TransactionDT` for IEEE-CIS). Scaling and imputation statistics are fitted on the training partition only.
- IEEE-CIS uses a chronological **prefix** of 300,000 rows for baselines and explanation quality and 150,000 for the imbalance experiment, taken as a prefix rather than a random draw so the time ordering survives.
- Columns more than 90% empty are dropped (two of 392). Remaining gaps are filled with training-partition medians for the random forest; XGBoost consumes them natively.
- Hyperparameters are fixed a priori and never tuned.

## 3. Resampling procedures

Resampling is applied to the **training partition only**, after the chronological split, so no synthetic or duplicated row can reach the test set.

| Regime | Procedure |
|---|---|
| natural | No resampling. Imbalance is addressed by instance weighting: `class_weight='balanced'` for logistic regression, `balanced_subsample` for the random forest, `scale_pos_weight = negatives/positives` for XGBoost. |
| undersample | `imblearn.under_sampling.RandomUnderSampler`, default (majority reduced to minority count), seeded. |
| smote | `imblearn.over_sampling.SMOTE(k_neighbors=5)`, seeded. |

The baselines use instance weighting rather than resampling deliberately: resampling is the manipulated variable of the imbalance experiment, and using it in the baseline would confound treatment with control.

## 4. Evaluation protocol

**Detection.** AUC-PR is the primary threshold-free measure; ROC-AUC is reported alongside; accuracy is never reported, because at a 0.172% base rate predicting the majority class scores 99.83%.

**Alert budget.** At budget *b* the alerts are the top *b* fraction of test transactions by model score, so *k* = ⌊*b·n*⌋ is fixed and the model determines only which cases fill the slots. Budgets are 0.1%, 0.5% and 1%. Because *k* is fixed, precision and recall at a budget are arithmetically linked (*precision = recall·F/k*) and are not independent evidence; both are reported because practitioners need both.

**Explanations.** Generated for two populations that answer different questions: the alert queue (the highest-scored test transactions, what an analyst sees) and a random sample from the remainder (the population the model scores). SHAP uses the exact algorithm for each model class — LinearExplainer for logistic regression, TreeExplainer for the ensembles — never KernelExplainer.

**LIME.** The locality kernel width is selected **per instance** by maximising the surrogate's local R², and instances whose best surrogate falls below the R² floor are excluded and counted. This is necessary rather than optional: at 390 features the default width gives a median local R² of 0.006 and returns numerical noise without raising an error. The same procedure applied to ULB selects the default width for most instances, so one implementation serves both dimensionalities.

**Inference.** 95% percentile bootstrap intervals over 2,000 resamples on every point estimate; Wilcoxon signed-rank for paired comparisons. At n = 400 every p-value is ≈1e-67 and carries no information, so Cohen's *d* and the number of top-5 features that change identity carry the argument.

**Metric validation.** The measures are validated on synthetic models whose true importances are known by construction, including a control that a random attribution must fail. See `experiments/01_synthetic_validation.py`.

## 5. Reproducing the results

Scripts are numbered in dependency order. From a clean checkout:

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python.exe experiments/00_get_data.py        # Kaggle API
.venv/Scripts/python.exe experiments/01_synthetic_validation.py
.venv/Scripts/python.exe experiments/02_baselines_ulb.py
.venv/Scripts/python.exe experiments/03_explanation_quality_ulb.py
.venv/Scripts/python.exe experiments/04_imbalance_explanation_ulb.py
.venv/Scripts/python.exe experiments/05_baselines_ieee.py
.venv/Scripts/python.exe experiments/06_explanation_quality_ieee.py
.venv/Scripts/python.exe experiments/08_exploratory_analysis.py
.venv/Scripts/python.exe experiments/10_imbalance_explanation_ieee.py
.venv/Scripts/python.exe experiments/12_stability_seeds.py
.venv/Scripts/python.exe experiments/13_lime_ieee.py
.venv/Scripts/python.exe experiments/14_fidelity_random_ieee.py
.venv/Scripts/python.exe experiments/15_cross_data_analysys.py
.venv/Scripts/python.exe experiments/16_stability_mechanism.py
.venv/Scripts/python.exe experiments/17_stability_mechanism_extended.py
.venv/Scripts/python.exe experiments/09_figures.py
.venv/Scripts/python.exe experiments/99_summary_tables.py
```

All randomness is seeded. The slowest step is `06_explanation_quality_ieee.py` at roughly five hours of processor time, dominated by the 1,000-instance stability loop over a 390-feature forest. Datasets are **not** redistributed: `00_get_data.py` fetches them, and `data/`, `artifacts/`, `logs/` and credentials are git-ignored.

## 6. How this methodology differs from earlier research

| Aspect | Common practice in the reviewed literature | This work |
|---|---|---|
| Object of measurement | The model; explanations are presented but not evaluated | The explanation output itself, with the model as instrument |
| Evidence for explanation quality | Predictive performance used as a proxy (>80% of 49 reviewed studies) | Fidelity, stability and agreement measured directly, with bootstrap intervals |
| Train/test split | Random | Chronological, because fraud drifts several-fold across deciles in both datasets |
| Operating point | A probability threshold, or none stated | A fixed alert budget reflecting a team's review capacity |
| Headline metric | Accuracy or ROC-AUC | AUC-PR plus precision and recall at budget; accuracy never reported |
| Imbalance handling | SMOTE applied by default, effect on explanations unexamined | Resampling treated as a manipulated variable, with performance held as a covariate and a seed control |
| Metric trustworthiness | Asserted | Validated on synthetic controls with known importances, including a random-attribution control |
| Explainer configuration | Library defaults | LIME's kernel width selected per instance by surrogate fit, with poorly-fitted instances excluded and counted |
| Reported conditions | Rarely stated | Seed, instance counts, curve length and top-k written into every results file |
| Agreement reporting | Rank correlation alone where reported | Rank correlation **and** top-k overlap, because they diverge (ρ = 0.553 with Jaccard 0.013 on IEEE-CIS) |

## 7. Limitations

- **Proxy properties, not usefulness.** Fidelity, stability and agreement are necessary but not sufficient for a useful explanation. Establishing usefulness requires professional fraud analysts, which was not feasible here.
- **IEEE-CIS subsample.** Results use a chronological prefix, so they strictly concern the dataset's first 85 days.
- **Untuned hyperparameters.** The models are reasonable, not optimal; tuning was avoided because its interaction with explanation quality is itself unmeasured.
- **One architecture in the imbalance experiment.** Generalisation across model families is not established.
- **No intervals on budget-level rates.** Precision and recall at a budget are reported as point estimates.
- **Seed stability covers one dataset.** 10 seeds on ULB only.
- **Rubric study outstanding.** The qualitative half of the actionability objective needs a second independent rater before inter-rater agreement can be reported.
