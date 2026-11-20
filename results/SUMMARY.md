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

An alert budget of *b* means the investigator reviews the top *b* fraction of test transactions, ranked by model score.

| Dataset | Model | AUC-PR | ROC-AUC | Precision @0.5% | Recall @0.5% |
|---|---|---|---|---|---|
| ULB | LogisticRegression | 0.7677 | 0.9796 | 0.218 | 0.861 |
| ULB | RandomForest | 0.8196 | 0.9529 | 0.211 | 0.833 |
| ULB | XGBoost | 0.8050 | 0.9773 | 0.211 | 0.833 |
| IEEE-CIS | LogisticRegression | 0.4139 | 0.8384 | 0.913 | 0.109 |
| IEEE-CIS | RandomForest | 0.5154 | 0.8845 | 0.980 | 0.117 |
| IEEE-CIS | XGBoost | 0.5756 | 0.8882 | 0.984 | 0.118 |

## S2 — explanation quality

Fidelity is reported on both populations because they behave differently. On ULB the random sample sits near zero because the model scores ~1e-6 there and deletion has no probability mass to remove. On IEEE-CIS the random sample scores 0.05-0.34, so that explanation does not apply: both tree ensembles score NEGATIVE with intervals excluding zero, meaning deleting the features SHAP ranked highest moves the fraud score up.

| Dataset | Model | Fidelity (alerts) | Fidelity (random) | Max-sensitivity | SHAP vs LIME ρ | Jaccard@5 |
|---|---|---|---|---|---|---|
| ULB | LogisticRegression | 0.645 [0.622, 0.669] | 0.032 [0.024, 0.040] | 0.039 [0.038, 0.039] | 0.651 [0.629, 0.672] | 0.393 [0.346, 0.440] |
| ULB | RandomForest | 0.170 [0.145, 0.196] | -0.000 [-0.000, 0.000] | 0.061 [0.057, 0.064] | 0.511 [0.484, 0.536] | 0.483 [0.446, 0.521] |
| ULB | XGBoost | 0.218 [0.182, 0.255] | -0.000 [-0.000, -0.000] | 0.123 [0.117, 0.130] | 0.309 [0.282, 0.337] | 0.266 [0.243, 0.289] |
| IEEE-CIS | LogisticRegression | 0.127 [0.111, 0.145] | 0.011 [-0.005, 0.028] | 0.038 [0.038, 0.038] | 0.413 [0.407, 0.420] | 0.107 [0.089, 0.125] |
| IEEE-CIS | RandomForest | 0.155 [0.146, 0.163] | -0.052 [-0.059, -0.045] | 0.204 [0.199, 0.210] | 0.399 [0.377, 0.419] | 0.122 [0.103, 0.145] |
| IEEE-CIS | XGBoost | 0.113 [0.102, 0.124] | -0.100 [-0.115, -0.084] | 0.960 [0.932, 0.990] | 0.231 [0.216, 0.243] | 0.050 [0.038, 0.062] |

### Was LIME's local surrogate usable at 390 features?

LIME weights perturbations by exp(-D^2/w^2) with w defaulting to sqrt(d)*0.75 = 14.81. At the head of the alert queue the model's response surface is locally flat, so a narrow kernel draws only perturbations with identical output and the regression returns noise. The width is therefore chosen per instance by maximising the surrogate's local R^2 over multipliers [1, 2, 3, 4]; instances whose best surrogate explains less than 50% of local variance are excluded. The default-width column shows what copying ULB's settings would have produced.

| Model | Local R² (tuned) | Local R² (default width) | Met R² floor | Widths chosen |
|---|---|---|---|---|
| LogisticRegression | 0.826 | 0.006 | 120 of 120 | 1x 7, 2x 69, 3x 34, 4x 10 |
| RandomForest | 0.995 | 0.010 | 120 of 120 | 1x 40, 2x 69, 3x 9, 4x 2 |
| XGBoost | 0.998 | 0.010 | 120 of 120 | 1x 36, 2x 73, 3x 9, 4x 2 |

### Do different models explain the same alert the same way?

H0: the two models rank the alert's features identically. Rows are ordered by effect size, not by significance: with n = 400 every p-value is around 1e-67, which states only that the sample is large, so it is reported last. **Top-5 changed** is how many of the five features an analyst would read differ between the two models, which is the quantity with an operational meaning. It is derived exactly from Jaccard rather than estimated: with both top sets of size k = 5, J = i/(10-i) for an intersection of size i, so i = 10J/(1+J).

| Dataset | Pair | Cohen's d | Top-5 changed | Spearman | Jaccard@5 | Wilcoxon p |
|---|---|---|---|---|---|---|
| ULB | LogisticRegression vs XGBoost | 4.92 | 3.02 of 5 | 0.290 [0.277, 0.304] | 0.246 [0.234, 0.259] | 1.36e-67 |
| ULB | LogisticRegression vs RandomForest | 3.70 | 2.52 of 5 | 0.352 [0.335, 0.369] | 0.329 [0.309, 0.349] | 1.36e-67 |
| ULB | RandomForest vs XGBoost | 3.30 | 2.22 of 5 | 0.529 [0.514, 0.543] | 0.385 [0.371, 0.399] | 1.36e-67 |
| IEEE-CIS | LogisticRegression vs XGBoost | 12.41 | 4.48 of 5 | 0.313 [0.308, 0.318] | 0.055 [0.048, 0.063] | 1.37e-67 |
| IEEE-CIS | LogisticRegression vs RandomForest | 9.94 | 4.87 of 5 | 0.553 [0.548, 0.557] | 0.013 [0.009, 0.017] | 1.37e-67 |
| IEEE-CIS | RandomForest vs XGBoost | 9.86 | 3.22 of 5 | 0.557 [0.552, 0.561] | 0.216 [0.206, 0.227] | 1.37e-67 |

## S3 — does resampling change the explanation?

One architecture, one test set, three training regimes, run independently on each dataset.

### ULB

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

### IEEE-CIS

150,000 of 590,540 rows (chronological, memory constrained).


| Regime | AUC-PR | ρ vs natural | Jaccard@5 vs natural |
|---|---|---|---|
| natural | 0.5100 | — | — |
| undersample | 0.5213 | 0.703 [0.699, 0.707] | 0.359 [0.343, 0.374] |
| smote | 0.5724 | 0.638 [0.633, 0.643] | 0.332 [0.317, 0.347] |

AUC-PR spread across regimes: **0.0624**

| Comparison | Wilcoxon p | mean ρ | Cohen's d |
|---|---|---|---|
| natural vs undersample | 1.37e-67 | 0.703 | 7.32 |
| natural vs smote | 1.37e-67 | 0.638 | 7.29 |

**Most-named feature per regime**

| Regime | Top features |
|---|---|
| natural | TransactionDT (63%), C1 (27%), V317 (8%) |
| undersample | V258 (55%), C2 (22%), TransactionAmt (21%) |
| smote | C1 (65%), M4 (18%), V258 (15%) |

H0 (rankings invariant to regime): **REJECTED**

## S5 — is the explanation stable across retraining seeds?

A control for S3. S2's max-sensitivity perturbs the *input*; this perturbs the *seed*, holding the data fixed. 10 seeds, 400 alerts, reference seed 42.

| Seed | ρ vs reference | Jaccard@5 | Cohen's d |
|---|---|---|---|
| 7 | 0.882 [0.877, 0.887] | 0.743 [0.725, 0.761] | 2.41 |
| 13 | 0.876 [0.872, 0.881] | 0.710 [0.692, 0.728] | 2.47 |
| 21 | 0.875 [0.871, 0.881] | 0.696 [0.679, 0.712] | 2.46 |
| 34 | 0.877 [0.872, 0.882] | 0.746 [0.728, 0.764] | 2.35 |
| 55 | 0.866 [0.861, 0.872] | 0.692 [0.673, 0.711] | 2.49 |
| 89 | 0.851 [0.845, 0.857] | 0.700 [0.681, 0.718] | 2.40 |
| 101 | 0.874 [0.869, 0.879] | 0.700 [0.680, 0.718] | 2.32 |
| 202 | 0.873 [0.868, 0.878] | 0.699 [0.681, 0.717] | 2.48 |
| 303 | 0.882 [0.878, 0.887] | 0.737 [0.719, 0.755] | 2.43 |

Mean Spearman across seeds: **0.873**

AUC-PR spread across seeds: **0.0134**

**The comparison that matters.** Changing the training seed moves the ranking to ρ = 0.873; changing the resampling regime moves it to ρ = 0.446. The divergence in S3 is therefore caused by the imbalance strategy, not by stochastic training.

## Objective 4 — can an investigator act on the explanation?

IEEE-CIS has 390 features, of which only **23** are semantically meaningful. ULB has none beyond Time and Amount.

| Model | Top-1 feature is semantic | Top-5 semantic (mean) |
|---|---|---|
| LogisticRegression | 0.2% | 0.4% |
| RandomForest | 2.0% | 5.5% |
| XGBoost | 2.5% | 4.4% |
