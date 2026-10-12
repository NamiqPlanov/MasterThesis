# Explainable AI for Fraud Detection in Financial Transactions

MSc thesis · Namig Planov · School of IT and Engineering, ADA University
Supervisor: Abdul Razaque

This repository holds the code and results for the thesis. The research question
is not whether machine learning can detect fraud — it can — but whether the
**explanations** such systems produce can withstand direct measurement.

A literature review of the field (Progress Report 2) found that over 80% of
studies use the model's predictive accuracy as a proxy for the quality of its
explanations, although the two are logically independent; that the two studies
which did test explanations on real fraud analysts found little or no benefit;
and that the field's dominant benchmark has anonymised features, so its
explanations cannot be acted on. This work measures the explanations instead of
assuming them.

## Status

| Stage | Objective | State |
|---|---|---|
| S1 | Baseline models, time-ordered splits | ULB done · IEEE-CIS pending |
| S2 | Fidelity, stability, explainer agreement | ULB done · IEEE-CIS pending |
| S3 | Imbalance × explanation experiment | pending |
| S4 | Metric validation on synthetic controls | **done — 7/7 checks pass** |
| S5 | Semantic actionability study | pending |

## Reproducing

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Fetching the data needs a Kaggle account. Save your API token to
`~/.kaggle/access_token`, and accept the
[IEEE-CIS competition rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules)
on the site first — the API refuses competition data until you do.

```bash
.venv\Scripts\python.exe experiments\00_get_data.py            # download
.venv\Scripts\python.exe experiments\01_synthetic_validation.py # S4 control
.venv\Scripts\python.exe experiments\02_baselines_ulb.py        # S1
.venv\Scripts\python.exe experiments\03_explanation_quality_ulb.py  # S2
```

Every script fixes its random seed and writes a JSON file to `results/`.

**Run the S4 control first.** It checks the metric implementations against
importances planted by construction. A metric that cannot recover a known
answer cannot be trusted on an unknown one, so if that script fails, nothing
downstream of it means anything.

## Layout

```
src/xai_metrics.py    fidelity, stability, agreement, bootstrap CIs
experiments/          numbered scripts, each a stage of the proposal
results/              JSON output, committed
data/                 downloaded datasets — NOT committed
artifacts/            fitted models and splits — NOT committed
```

## Results so far

**S1 — ULB, time-ordered split** (199,364 train / 85,443 test, 0.126% fraud in test)

| Model | AUC-PR | ROC-AUC | Recall @0.5% budget |
|---|---|---|---|
| Logistic Regression | 0.7677 | 0.9796 | 0.861 |
| Random Forest | **0.8196** | 0.9529 | 0.833 |
| XGBoost | 0.8050 | 0.9773 | 0.833 |

Accuracy is not reported: predicting "legitimate" for everything scores 99.874%
on this test set and catches nothing.

**S2 — explanation quality**, 1,000 instances (400 alert queue, 600 random):

- SHAP and LIME, explaining the *same* model and the *same* prediction, agree on
  only 26–48% of their top-5 features (Jaccard@5). On XGBoost, Spearman is
  0.301 [0.273, 0.327].
- Three models within 0.04 AUC-PR of one another produce substantially
  different explanations (cross-model Jaccard@5 0.25–0.39).
- XGBoost names `V14` — an anonymised principal component — in 76% of alerts.
- Fidelity is only measurable where the model is actually deciding something:
  it drops to ≈0 on randomly drawn legitimate transactions.

## Data

Downloaded, not redistributed.

- **ULB Credit Card Fraud Detection** — 284,807 transactions, 492 fraudulent
  (0.172%); features V1–V28 are PCA components. Open Database License;
  contents under the Database Contents License.
- **IEEE-CIS Fraud Detection** (Vesta Corporation) — 590,540 transactions,
  ~3.5% fraud, several hundred semantically named features. Governed by the
  Kaggle competition rules.

## Licence

Code: MIT. Results and text: © Namig Planov.
