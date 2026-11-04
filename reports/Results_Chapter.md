# Chapter 4: Results

Every number in this chapter is read from the JSON files written by the experiment scripts
and collected in `results/SUMMARY.md`. Where a result changed a conclusion during the work,
or where an experimental condition had to be corrected, that is reported in place rather
than silently resolved, because the methodological point of the study is that unexamined
measurement practice produces unreliable claims.

## 4.1 Validation of the measurement apparatus

The explanation-quality measures were validated before use on synthetic models whose true
feature importances are known by construction. All seven checks pass: `LinearExplainer`
reproduces the closed-form attribution to within 10⁻⁶; both logistic regression and the
random forest recover the five planted informative features (ρ > 0.9 and ρ > 0.8
respectively); the logistic-regression global top-5 matches the planted top-5 exactly
(Jaccard = 1.0); SHAP outperforms a random attribution on faithfulness while a random
attribution fails recovery (|ρ| < 0.3); and the stability measure is finite and orders the
models as expected, with logistic regression smoother than the forest.

These are weak tests in the sense that passing them is necessary rather than sufficient.
Their value is that failure would have invalidated everything downstream, and two of them —
the random-attribution control and the closed-form check — are the kind of test the
literature reviewed in Chapter 2 does not report.

## 4.2 Detection baselines and the alert budget

### 4.2.1 Baseline performance

| Dataset | Model | AUC-PR | ROC-AUC | Precision @0.5% | Recall @0.5% |
|---|---|---|---|---|---|
| ULB | Logistic regression | 0.7677 | 0.9796 | 0.218 | 0.861 |
| ULB | Random forest | 0.8196 | 0.9529 | 0.211 | 0.833 |
| ULB | XGBoost | 0.8050 | 0.9773 | 0.211 | 0.833 |
| IEEE-CIS | Logistic regression | 0.4139 | 0.8384 | 0.913 | 0.109 |
| IEEE-CIS | Random forest | 0.5154 | 0.8845 | 0.980 | 0.117 |
| IEEE-CIS | XGBoost | 0.5756 | 0.8882 | 0.984 | 0.118 |

These figures sit below the values commonly published for both datasets, and the reason is
the chronological split required by §3.2.2 rather than any deficiency in the models.
Fraud is not stationary in either dataset: measured across chronological deciles the fraud
rate varies several-fold (Figure 1), so a model evaluated on a random split is permitted to
learn from the future. The gap between these numbers and the literature's is therefore an
estimate of how much the field's standard protocol flatters itself.

### 4.2.2 Precision and recall expose opposite failure modes

The two datasets fail in opposite directions at the same alert budget, and neither AUC-PR
nor recall alone would reveal it. At b = 0.5%, logistic regression on ULB reviews 427
transactions, catches 93 of 108 frauds (recall 0.861) and is wrong about four alerts in
five (precision 0.218). XGBoost on IEEE-CIS reviews 450 transactions, is right about
almost every one (precision 0.984) and misses 3,316 of 3,759 frauds (recall 0.118).

The operational readings are entirely different. The ULB configuration describes a team
that catches most fraud at the cost of disturbing a large number of legitimate customers.
The IEEE-CIS configuration describes a team whose every alert is worth opening and whose
budget is far too small for the fraud present — at a 3.37% base rate, a 0.5% budget cannot
physically contain more than about a seventh of the fraud, so recall is capacity-bound
rather than model-bound. Reporting recall alone would make ULB look seven times better;
reporting precision alone would reverse the ranking.

As noted in §3.4.2 these two rates are arithmetically linked at fixed *k* and are not
independent evidence. They are reported together because each answers a question a
practitioner asks, not because they constitute two tests.

### 4.2.3 Threshold-free ranking disagrees with the operating point

On ULB the random forest attains the best AUC-PR (0.8196) while logistic regression attains
both the best precision (0.218 against 0.211) and the best recall (0.861 against 0.833) at
the 0.5% budget. The threshold-free average over all operating points therefore ranks the
models differently from the single operating point a team would deploy.

The margins are small — 0.007 in precision, 0.028 in recall — and no confidence intervals
were computed for the budget-level rates, so this should be read as a demonstration that
the two kinds of measure *can* disagree rather than as evidence that logistic regression is
the better model here. The methodological consequence stands regardless: a study that
reports AUC-PR alone has not established which model a team should run.

## 4.3 Fidelity

| Dataset | Model | Fidelity, alert queue | Fidelity, random sample |
|---|---|---|---|
| ULB | Logistic regression | 0.645 [0.622, 0.669] | 0.032 [0.024, 0.040] |
| ULB | Random forest | 0.170 [0.145, 0.196] | −0.000 [−0.000, 0.000] |
| ULB | XGBoost | 0.218 [0.182, 0.255] | −0.000 [−0.000, −0.000] |
| IEEE-CIS | Logistic regression | 0.127 [0.111, 0.145] | 0.011 [−0.005, 0.028] |
| IEEE-CIS | Random forest | 0.155 [0.146, 0.163] | **−0.052** [−0.059, −0.045] |
| IEEE-CIS | XGBoost | 0.113 [0.102, 0.124] | **−0.100** [−0.115, −0.084] |

Three patterns matter, and the third was not anticipated.

First, the interpretable model is the faithful one, and by a wide margin on ULB. Logistic
regression's deletion fidelity of 0.645 means that removing the twenty features SHAP named
pulls the fraud score down by two thirds of its starting value. For the tree ensembles the
same operation moves it by 0.17–0.22. SHAP is exact for all three model classes here, so
this is not explainer error; it reflects that attributions for an additive model identify
the features the model genuinely uses, while an ensemble's prediction is distributed across
interactions that no single-feature deletion dislodges.

Second, fidelity on the alert queue is far higher than on the random sample in every
configuration. On ULB this is largely mechanical: a legitimate ULB transaction is scored
near 10⁻⁶, so there is almost no probability mass for deletion to remove and the measure
has little room to register anything. The practical implication is that fidelity figures
quoted over a random sample of ULB transactions describe a regime where nothing can move,
and are not evidence about the alerts an analyst reads.

Third — and this is where the mechanical explanation fails — the same reasoning does not
transfer to IEEE-CIS, and measuring it rather than assuming it produced a stronger result.
The IEEE-CIS random sample is *not* scored near zero: median predicted scores are 0.339,
0.088 and 0.045 for the three models, so there is ample probability mass available. Yet
both tree ensembles return **negative** deletion fidelity, with confidence intervals that
exclude zero: −0.052 [−0.059, −0.045] for the random forest and −0.100 [−0.115, −0.084] for
XGBoost.

A negative deletion fidelity means that removing the twenty features the attribution ranked
most important moves the fraud score in the *wrong direction* — it rises. On ordinary
transactions, SHAP's ranking for these two models is not merely uninformative about what
the model is doing; it is anti-correlated with it. Logistic regression does not show the
effect (0.011, interval straddling zero), so this is a property of the ensembles rather than
of the measure.

The practical consequence is narrower than it first appears, and should not be overstated:
these are transactions the model is not flagging, so no investigator would ordinarily read
their explanations. The consequence is for *global* explanation. A SHAP summary plot
aggregated over a representative sample of transactions — the standard artefact in the
literature reviewed in Chapter 2 — is dominated by exactly this population, where the
per-instance attributions demonstrably point the wrong way.

This measurement existed on ULB but not on IEEE-CIS in the first version of these
experiments, because the IEEE-CIS script measured the alert queue only. It was added when
the asymmetry was noticed while checking experimental conditions, which is the second
occasion on which that check changed a result rather than merely confirming one.

## 4.4 Stability

### 4.4.1 Input stability, and a conclusion that changed with sample size

| Dataset | Model | Max-sensitivity (n = 1,000) | Earlier value (n = 80) |
|---|---|---|---|
| ULB | Logistic regression | 0.039 [0.038, 0.039] | 0.039 |
| ULB | Random forest | 0.061 [0.057, 0.064] | 0.087 [0.072, 0.104] |
| ULB | XGBoost | 0.123 [0.117, 0.130] | 0.085 [0.064, 0.112] |
| IEEE-CIS | Logistic regression | 0.038 [0.038, 0.038] | — |
| IEEE-CIS | Random forest | 0.204 [0.199, 0.210] | — |
| IEEE-CIS | XGBoost | 0.960 [0.932, 0.990] | 0.603 [0.574, 0.634] |

Lower is better; the quantity is the largest relative change in the attribution vector
when the input is nudged within an ε-ball of 5% of each feature's magnitude.

The right-hand column records a methodological failure worth reporting. An earlier version
of these experiments sampled 80 instances for this metric where the proposal commits to at
least 1,000, and no script recorded the sample size, so the discrepancy was invisible in
the results files. At 80 instances the two ULB tree models had overlapping intervals —
0.087 [0.072, 0.104] and 0.085 [0.064, 0.112] — and the natural conclusion was that they
were comparably unstable. At 1,000 instances the intervals are disjoint and XGBoost is
twice as unstable as the random forest. On IEEE-CIS the XGBoost estimate moved further
still, from 0.603 to 0.960. The conclusion reversed because of the sample size, not because
of anything about the models, and the only reason it was caught is that §3.9's
condition-recording was added in response to the supervisor's review.

The substantive findings are these. Logistic regression is stable and essentially
dimension-independent (0.039 and 0.038 on 30 and 390 features). The tree ensembles are not,
and they degrade sharply with dimensionality: the random forest moves from 0.061 to 0.204
and XGBoost from 0.123 to 0.960. A max-sensitivity of 0.960 means that a 5% perturbation of
the input — a change far smaller than the difference between two similar transactions —
changes the attribution vector by almost as much as the vector's own magnitude. On
IEEE-CIS, XGBoost's explanations are approximately as large as their own instability. They
cannot support an investigator's decision about an individual alert, and XGBoost is the
best-performing detector on that dataset (AUC-PR 0.5756).

That conjunction is the study's sharpest result: on IEEE-CIS the most accurate model is the
one whose explanations are least able to survive a negligible change in input.

### 4.4.2 Training stability across seeds

Ten XGBoost models were fitted on ULB differing only in random seed, and each explained the
same fixed alert queue. Agreement with the reference seed is high and tightly bounded:
per-seed Spearman ranges from 0.851 [0.845, 0.857] to 0.882 [0.878, 0.887], with a mean of
**0.873**, and Jaccard@5 from 0.692 to 0.746. AUC-PR varies across the ten seeds by only
0.0134.

Explanations are therefore mostly, but not entirely, a property of the data rather than of
one training run. The Wilcoxon test rejects exact invariance for every seed (p < 0.05,
Cohen's *d* between 2.32 and 2.49), which is the expected outcome of testing an exact null
against 400 paired instances and should not be read as a substantive finding. The
substantive reading is the magnitude: a quarter of the top-five features change identity
between two runs of the same recipe on the same data, which is a caveat on any claim that a
particular feature "drives" fraud detection.

The purpose of this experiment is comparative, and §4.6 uses it accordingly.

## 4.5 Agreement between explanations

### 4.5.1 SHAP against LIME

| Dataset | Model | Spearman | Jaccard@5 | Local *R*² (tuned) | Local *R*² (default *w*) |
|---|---|---|---|---|---|
| ULB | Logistic regression | 0.651 [0.629, 0.672] | 0.393 [0.346, 0.440] | 0.835 | 0.835 |
| ULB | Random forest | 0.511 [0.484, 0.536] | 0.483 [0.446, 0.521] | 0.980 | 0.969 |
| ULB | XGBoost | 0.309 [0.282, 0.337] | 0.266 [0.243, 0.289] | 0.974 | 0.949 |
| IEEE-CIS | Logistic regression | 0.413 [0.407, 0.420] | 0.107 [0.089, 0.125] | 0.826 | 0.006 |
| IEEE-CIS | Random forest | 0.399 [0.377, 0.419] | 0.122 [0.103, 0.145] | 0.995 | 0.010 |
| IEEE-CIS | XGBoost | 0.231 [0.216, 0.243] | 0.050 [0.038, 0.062] | 0.998 | 0.010 |

Two explainers applied to the same model and the same transaction disagree substantially
everywhere, and the disagreement worsens with model complexity: on ULB, ρ falls from 0.651
for logistic regression to 0.309 for XGBoost, and the same ordering holds on IEEE-CIS. On
IEEE-CIS the top-five overlap is 0.05–0.12, meaning two standard explainers describing one
prediction share at most one feature in eight of the five an analyst would read.

#### The measurement very nearly produced a false result

The final two columns record a methodological finding that is arguably more important than
the agreement figures themselves, and it is reported here because this study's subject is
the reliability of explanation measurement.

LIME weights each sampled perturbation by exp(−*D*²/*w*²), with *w* defaulting to
√*d* · 0.75. On IEEE-CIS the first version of this experiment copied ULB's settings
verbatim — ostensibly the consistent thing to do — and reported SHAP–LIME correlations of
0.294 to 0.651. Those numbers were meaningless. At *d* = 390 the default width gives a
median local *R*² of **0.006 to 0.010**: LIME's surrogate explained essentially none of the
model's local behaviour, its returned coefficients were numerical noise of order 10⁻²⁹ in
the worst cases, and about a third of instances produced attribution vectors so uniformly
near zero that a rank correlation against them was mathematically undefined. The reported
correlations were computed over that noise.

The cause is a flat response surface. At the head of the alert queue the model's predicted
probability is approximately 0.99996, so a narrow kernel draws only perturbations whose
output is identical, the weighted regression target has no variance, and the fit collapses.
Raising the sample count does not help — *R*² remains near 0.001 at 10,400 samples — because
the problem is the weighting, not the sampling.

Three facts determined the fix. Widening the kernel repairs the fit at the head of the
queue. But the required width is a property of the *instance*, not the dataset: across queue
positions whose predicted probabilities differ only in the fifth decimal, the best
multiplier falls from 2× at positions 0–11 to 1× at positions 200–211, where widening
instead destroys locality and drops *R*² from 0.97 to 0.51. And a width chosen on held-out
instances is therefore actively misleading — an intermediate version of this script selected
1× on positions 200–259 and then reported a median *R*² of 0.010 on the positions it
actually measured.

The width is consequently selected per instance by maximising the surrogate's local *R*², a
criterion that depends only on the LIME fit and never consults the SHAP vector it is later
compared against, with instances below *R*² = 0.5 excluded and counted. This lifts median
*R*² to 0.826–0.998 and admits all 120 instances for every model on IEEE-CIS.

Applying the identical procedure to ULB is what makes the two datasets comparable, and the
result validates the design: on ULB the search selects the default width for 99–102 of 120
instances per model and leaves median *R*² unchanged to three decimals for logistic
regression. The same code adapts to 30 features and to 390 rather than being tuned for one
and imposed on the other. ULB's agreement figures shift only slightly as a result — ρ for
logistic regression moves from 0.701 to 0.651, chiefly through the *R*² floor excluding a
handful of poorly-fitted instances — while the random forest and XGBoost are unchanged to
two decimals.

The general lesson is that **matching a hyperparameter across conditions is not the same as
matching the experimental condition**. Copying LIME's kernel formula made the two
experiments look identical while leaving them incomparable in substance, and it would have
produced a publishable-looking table of correlations with no information in it. The
condition that has to be held constant is that the surrogate actually fits, and verifying
that requires reporting a fit statistic the literature reviewed in Chapter 2 does not report.

### 4.5.2 Different models, same alert

| Dataset | Pair | Spearman | Jaccard@5 | Cohen's *d* |
|---|---|---|---|---|
| ULB | LR vs RF | 0.352 [0.335, 0.369] | 0.329 [0.309, 0.349] | 3.70 |
| ULB | LR vs XGB | 0.290 [0.277, 0.304] | 0.246 [0.234, 0.259] | 4.92 |
| ULB | RF vs XGB | 0.529 [0.514, 0.543] | 0.385 [0.371, 0.399] | 3.30 |
| IEEE-CIS | LR vs RF | 0.553 [0.548, 0.557] | 0.013 [0.009, 0.017] | 9.94 |
| IEEE-CIS | LR vs XGB | 0.313 [0.308, 0.318] | 0.055 [0.048, 0.063] | 12.41 |
| IEEE-CIS | RF vs XGB | 0.557 [0.552, 0.561] | 0.216 [0.206, 0.227] | 9.86 |

Every Wilcoxon test returns p ≈ 1.4 × 10⁻⁶⁷. As §3.8 anticipated, this is a statement
about n = 400 and not about the models; the effect sizes carry the argument.

Three models trained on the same data, scoring the same transaction, and explained by the
exact SHAP method appropriate to each do not agree about why that transaction is
suspicious. On ULB the closest pair shares barely half the ranking (ρ = 0.529) and under
40% of its top five. This matters because the choice of model is normally justified on
predictive grounds — AUC-PR separates these three by 0.05 on ULB — while the explanation an
institution would place in a regulatory file, or read to a customer, depends heavily on
which of them was picked.

The IEEE-CIS rows contain the chapter's clearest illustration of why both agreement
measures are necessary. Logistic regression and the random forest reach ρ = 0.553, a
moderate rank correlation that a study reporting correlation alone would describe as
substantial agreement. Their Jaccard@5 is **0.013**: across 400 alerts the two models
almost never nominate a single shared feature among their top five. They order the long
tail of 390 mostly-irrelevant features similarly while disagreeing about everything an
analyst would read. Rank correlation over a high-dimensional attribution vector is
dominated by the features nobody will look at.

## 4.6 Does imbalance handling distort explanations?

| Dataset | Regime | AUC-PR | ρ vs natural | Jaccard@5 vs natural | Cohen's *d* |
|---|---|---|---|---|---|
| ULB | natural | 0.8050 | — | — | — |
| ULB | undersample | 0.7806 | 0.446 [0.432, 0.460] | 0.433 [0.418, 0.448] | 4.00 |
| ULB | SMOTE | 0.7992 | 0.581 [0.568, 0.596] | 0.502 [0.484, 0.522] | 2.94 |
| IEEE-CIS | natural | 0.5100 | — | — | — |
| IEEE-CIS | undersample | 0.5213 | 0.703 [0.699, 0.707] | 0.359 [0.343, 0.374] | 7.32 |
| IEEE-CIS | SMOTE | 0.5724 | 0.638 [0.633, 0.643] | 0.332 [0.317, 0.347] | 7.29 |

The null hypothesis that explanation rankings are invariant to the resampling regime is
rejected on both datasets.

On ULB the design works as intended. AUC-PR spans 0.0244 across the three regimes — a
difference of no practical consequence — while explanation agreement with the
natural-imbalance model falls to ρ = 0.446 under undersampling. The most-named feature
changes outright: the natural model attributes 88% of its alerts primarily to V14, the
undersampled model names V4 most often (39%) with V14 demoted to 27%. Two models a
practitioner would regard as interchangeable on detection quality tell materially different
stories about the same transactions.

On IEEE-CIS the covariate is held less well and this must be conceded. AUC-PR spans 0.0624,
with SMOTE improving on the natural regime from 0.5100 to 0.5724 — a 12% relative gain that
is not negligible. The explanations diverge there too (ρ = 0.638, *d* = 7.29), but part of
that divergence may reflect a genuinely better model rather than a distortion. The ULB
result, where performance is effectively flat, carries the argument; the IEEE-CIS result
replicates the direction without isolating the cause as cleanly.

### The control that makes this interpretable

The obvious objection is that SHAP attributions are simply noisy, and that any two models
would disagree this much. §4.4.2 answers it. Changing the training seed — same data, same
recipe, same architecture — moves the ranking only to ρ = 0.873. Changing the resampling
regime moves it to 0.446–0.581.

| Manipulation | ρ vs reference |
|---|---|
| Different training seed (same data, same recipe) | 0.873 |
| SMOTE oversampling (same seed, rebalanced) | 0.581 |
| Random undersampling (same seed, rebalanced) | 0.446 |

Stochastic training accounts for roughly a tenth of the disagreement that resampling
produces. The divergence in the imbalance experiment is attributable to the imbalance
strategy itself. This comparison is shown in Figure 5.

## 4.7 Semantic actionability

| Model | Top-1 feature is semantic | Top-5 semantic (mean) |
|---|---|---|
| Logistic regression | 0.2% | 0.4% |
| Random forest | 2.0% | 5.5% |
| XGBoost | 2.5% | 4.4% |

Of the 390 features retained on IEEE-CIS, 23 are semantically meaningful. ULB has none
beyond *Time* and *Amount*, so it scores zero by construction — which is the comparison
Objective 4 exists to make, and which already disqualifies the field's dominant benchmark
from any claim about actionable explanation.

The finding on IEEE-CIS is more damaging than the dataset's feature names suggest. Having
real feature names available is not sufficient: all three models rank an opaque feature
first for between 97.5% and 99.8% of alerts. The features that dominate are Vesta's
engineered columns and the counting families — XGBoost names `C1` first for 66% of its
alerts, logistic regression names `V201` for 35% — none of which an investigator can
pursue. A semantic feature space is a necessary condition for an actionable explanation,
and these models do not use it.

The qualitative half of Objective 4, which scores explanation reports against the rubric in
§3.6.4 using two independent raters, is outstanding. The quantitative result above does not
depend on it, but the objective is not complete without it.

## 4.8 Summary of findings

1. **Explanations are much less faithful for the accurate models.** Deletion fidelity on
   ULB is 0.645 for logistic regression against 0.170–0.218 for the tree ensembles (§4.3).
2. **Explanation stability collapses with dimensionality for tree ensembles.**
   Max-sensitivity rises from 0.123 to 0.960 for XGBoost between 30 and 390 features, while
   logistic regression is unchanged at ≈0.039 (§4.4.1).
3. **On IEEE-CIS the best detector has the least stable explanations** — XGBoost leads on
   AUC-PR at 0.5756 and records max-sensitivity 0.960 (§4.4.1).
4. **Three models agree on performance and disagree on reasons.** Cross-model Jaccard@5
   falls to 0.013 on IEEE-CIS despite ρ = 0.553 (§4.5.2).
5. **Resampling changes explanations without changing detection.** On ULB, AUC-PR spans
   0.0244 while agreement falls to ρ = 0.446; retraining noise accounts for only ρ = 0.873,
   so the effect is attributable to the resampling strategy (§4.6).
6. **A semantic feature space is not sufficient for actionability.** With 23 meaningful
   features available, models rank one first in 0.2–2.5% of alerts (§4.7).
7. **SHAP's attributions are anti-faithful off the alert queue.** On IEEE-CIS, deletion
   fidelity on a random sample is −0.052 and −0.100 for the two tree ensembles, intervals
   excluding zero, with ample probability mass available to move. Global SHAP summary plots
   are aggregated over exactly this population (§4.3).
8. **Two standard explainers disagree about the same prediction**, and worsen with model
   complexity: ρ falls from 0.651 to 0.309 across model classes on ULB, with top-five
   overlap of 0.05–0.12 on IEEE-CIS (§4.5.1).
9. **Measurement conditions must be recorded or conclusions are unsafe.** Three times in
   this work, checking a condition changed a result rather than confirming it: an
   unrecorded sample size of 80 instead of 1,000 reversed a conclusion about the relative
   stability of two models (§4.4.1); LIME's default locality kernel was found to return
   numerical noise at 390 features, invalidating a first set of agreement figures
   (§4.5.1); and an asymmetric fidelity measurement concealed the anti-faithfulness result
   above (§4.3).

---

*Outstanding at the time of writing:* the two-rater rubric study for Objective 4 (§4.7),
which requires a second independent rater and is the only item in this chapter that cannot
be completed computationally.
