# Results summary

Generated from the JSON written by each experiment. Do not edit by hand — re-run `experiments/99_summary_tables.py`.

## S4 — metric validation on synthetic controls

Checks that the measures recover importances planted by construction. If any of these fail, nothing downstream is trustworthy.

| Check | Result |
|---|---|
| LinearExplainer matches closed form (<1e-6) | PASS |
| LR ranks the 5 informative features correctly (rho > 0.9) | PASS |
| RF ranks the 5 informative features correctly (rho > 0.8) | PASS |
| LR global top-5 = planted top-5 (Jaccard = 1.0) | PASS |
| SHAP beats random on faithfulness | PASS |
| random attribution fails recovery (|rho| < 0.3) | PASS |
| stability finite and ordered (LR smoother than RF) | PASS |

## S1 — baseline models (time-ordered splits)

| Dataset | Model | AUC-PR | ROC-AUC | Recall @0.5% budget |
|---|---|---|---|---|
| ULB | LogisticRegression | 0.7677 | 0.9796 | 0.861 |
| ULB | RandomForest | 0.8196 | 0.9529 | 0.833 |
| ULB | XGBoost | 0.8050 | 0.9773 | 0.833 |
| IEEE-CIS | LogisticRegression | 0.4139 | 0.8384 | 0.109 |
| IEEE-CIS | RandomForest | 0.5154 | 0.8845 | 0.117 |
| IEEE-CIS | XGBoost | 0.5756 | 0.8882 | 0.118 |

## S2 — explanation quality

| Dataset | Model | Fidelity (alerts) | Max-sensitivity | SHAP vs LIME ρ | Jaccard@5 |
|---|---|---|---|---|---|
| ULB | LogisticRegression | 0.645 [0.622, 0.669] | 0.039 [0.038, 0.040] | 0.701 [0.684, 0.719] | 0.460 [0.412, 0.505] |
| ULB | RandomForest | 0.170 [0.145, 0.196] | 0.087 [0.072, 0.104] | 0.512 [0.486, 0.537] | 0.483 [0.445, 0.522] |
| ULB | XGBoost | 0.218 [0.182, 0.255] | 0.085 [0.064, 0.112] | 0.301 [0.273, 0.327] | 0.260 [0.237, 0.283] |
| IEEE-CIS | LogisticRegression | -0.196 [-0.229, -0.164] | 0.035 [0.034, 0.035] | — | — |
| IEEE-CIS | RandomForest | -0.369 [-0.380, -0.358] | 0.269 [0.251, 0.287] | — | — |
| IEEE-CIS | XGBoost | -0.823 [-0.835, -0.811] | 0.603 [0.574, 0.634] | — | — |

### Do different models explain the same alert the same way?

| Dataset | Pair | Spearman | Jaccard@5 |
|---|---|---|---|
| ULB | LogisticRegression vs RandomForest | 0.352 [0.335, 0.369] | 0.329 [0.309, 0.349] |
| ULB | LogisticRegression vs XGBoost | 0.290 [0.277, 0.304] | 0.246 [0.234, 0.259] |
| ULB | RandomForest vs XGBoost | 0.529 [0.514, 0.543] | 0.385 [0.371, 0.399] |
| IEEE-CIS | LogisticRegression vs RandomForest | 0.553 [0.548, 0.557] | 0.013 [0.009, 0.017] |
| IEEE-CIS | LogisticRegression vs XGBoost | 0.313 [0.308, 0.318] | 0.055 [0.048, 0.063] |
| IEEE-CIS | RandomForest vs XGBoost | 0.557 [0.552, 0.561] | 0.216 [0.206, 0.227] |

## S3 — does resampling change the explanation?

One architecture, one test set, three training regimes.

| Regime | AUC-PR | ρ vs natural | Jaccard@5 vs natural |
|---|---|---|---|
| natural | 0.8050 | — | — |
| undersample | 0.7806 | 0.446 [0.432, 0.460] | 0.433 [0.418, 0.448] |
| smote | 0.7992 | 0.581 [0.568, 0.596] | 0.502 [0.484, 0.522] |

AUC-PR spread across regimes: **0.0244**

| Comparison | Wilcoxon p | mean ρ | Cohen's d |
|---|---|---|---|
| natural vs undersample | 1.36e-67 | 0.446 | 4.00 |
| natural vs smote | 1.36e-67 | 0.581 | 2.94 |

**Most-named feature per regime**

| Regime | Top features |
|---|---|
| natural | V14 (88%), V4 (7%), Time (3%) |
| undersample | V4 (39%), V10 (33%), V14 (27%) |
| smote | V14 (58%), Time (30%), V11 (10%) |

H0 (rankings invariant to regime): **REJECTED**

## Objective 4 — can an investigator act on the explanation?

IEEE-CIS has 390 features, of which only **23** are semantically meaningful. ULB has none beyond Time and Amount.

| Model | Top-1 feature is semantic | Top-5 semantic (mean) |
|---|---|---|
| LogisticRegression | 0.2% | 0.4% |
| RandomForest | 2.0% | 5.5% |
| XGBoost | 2.5% | 4.4% |
