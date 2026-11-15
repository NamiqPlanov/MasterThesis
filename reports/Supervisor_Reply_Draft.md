# Draft message to Prof. Abdul Razaque

*Written 14 November 2026, after all five review points were completed. Plain
language on purpose. Edit before sending — especially the closing request,
which assumes you want him to either rate a sample himself or suggest someone.*

---

**Subject:** All five review points completed — results and next steps

Dear Professor Razaque,

Thank you for the feedback. I have completed all five points you raised. Below is
what I did and what came out of it.

## 1. The alert budget is now defined

An alert budget is the share of transactions an investigation team has the capacity
to review. At a budget of *b*, the alerts are the top *b* fraction of test
transactions ranked by model score — so the budget fixes how many cases get
reviewed, and the model decides only which cases fill them. I report three budgets
(0.1%, 0.5% and 1%) and use 0.5% wherever a single operating point is needed. At
0.5% that is 427 alerts on ULB and 450 on IEEE-CIS.

I chose a fixed capacity rather than a fixed probability threshold because a
threshold gives a different number of alerts every day, which is not how a real
team works.

## 2. Precision and recall are now reported

Both are reported at every budget. They showed something AUC-PR had hidden: the
two datasets fail in opposite directions.

| Dataset | Model | Precision @0.5% | Recall @0.5% |
|---|---|---|---|
| ULB | Logistic regression | 0.218 | 0.861 |
| IEEE-CIS | XGBoost | 0.984 | 0.118 |

On ULB the model catches 86% of fraud but is wrong about four alerts in five. On
IEEE-CIS almost every alert is genuine, but seven frauds in eight are missed.
Reporting only one of the two rates would have made these look similar.

A second point surfaced: on ULB the random forest has the best AUC-PR, while
logistic regression wins both precision and recall at the 0.5% budget. The margins
are small, so I present this as evidence that the two kinds of measure *can*
disagree, not as proof that one model is better.

## 3. Experimental conditions are now consistent

Every script now writes its conditions — seed, number of instances, curve length,
top-*k* — into its results file, so a mismatch cannot stay hidden. That change
immediately found two real problems.

**The stability metric was using 80 instances where the proposal commits to 1,000.**
Fixing it changed a conclusion. At 80 instances the two tree models on ULB looked
equally unstable; at 1,000 their confidence intervals no longer overlap and XGBoost
is twice as unstable as the random forest.

**The SHAP-versus-LIME comparison existed on ULB but not on IEEE-CIS.** Making it
consistent turned out to be the hardest part, and it produced a finding of its own.
LIME weights its samples by distance from the transaction, with a default width
based on the number of features. On ULB (30 features) this works well. On IEEE-CIS
(390 features) it fails completely: at the top of the alert queue the model's output
is so flat that all of LIME's samples return the same prediction, its internal
regression has nothing to fit, and the weights it returns are numerical noise — its
own goodness-of-fit is 0.006. Using more samples does not help. I now choose the
width separately for each transaction, based on how well LIME's local model fits,
and discard any instance where it still does not fit. This raises the fit to between
0.826 and 0.998.

So the condition I hold constant across the two datasets is that LIME's local model
must actually fit — not that both runs use the same formula, which would have made
the two experiments look matched while leaving them incomparable. Both datasets now
run the same code.

I also found that fidelity was measured on two populations on ULB but only one on
IEEE-CIS. Closing that gap produced the most surprising result of the project, in
section 5 below.

## 4. Stability across multiple runs is now measured

I treated this as two separate questions, because a method can pass one and fail the
other:

- **Perturbing the input, model fixed.** Max-sensitivity (lower is better):

| Model | ULB | IEEE-CIS |
|---|---|---|
| Logistic regression | 0.039 | 0.038 |
| Random forest | 0.061 | 0.204 |
| XGBoost | 0.123 | **0.960** |

  XGBoost's 0.960 means a 5% change to the input — smaller than the difference
  between two similar transactions — changes the explanation by almost as much as
  the explanation's own size. And XGBoost is the best detector on that dataset.

- **Perturbing the seed, data fixed.** I trained ten XGBoost models differing only
  in random seed and compared their explanations of the same alerts. Agreement is
  high: mean ρ = 0.873, with AUC-PR varying by only 0.0134.

The second one also gave me the control I needed for the imbalance experiment.
Retraining moves the explanation only to ρ = 0.873, while changing the resampling
method moves it to 0.45–0.58. So the divergence I reported earlier is caused by the
resampling strategy, not by random training noise. That was the obvious objection to
that result, and it is now answered.

## 5. Statistical evidence is attached to every comparison

Wilcoxon signed-rank tests and Cohen's *d* on all paired comparisons, with 95%
bootstrap confidence intervals on every estimate.

One honest caveat: with 400 paired instances every *p*-value comes out around
10⁻⁶⁷, which only tells us the sample is large. I report the *p*-values because the
proposal promised them, but I lead with the effect sizes, which range from 2.9 to
12.4 and carry the actual argument.

## Main results so far

- **Resampling changes the explanation without changing detection.** On ULB,
  AUC-PR varies by only 0.0244 across the three regimes while explanation agreement
  falls to ρ = 0.446 (*d* = 4.00). The most-named feature changes outright.
- **The best detector has the least stable explanations.** On IEEE-CIS, XGBoost
  leads on AUC-PR (0.5756) and is worst on stability (0.960).
- **SHAP's attributions point the wrong way away from the alert queue.** On
  IEEE-CIS, deleting the features SHAP ranked highest makes the fraud score go
  *up* for both tree models (−0.052 and −0.100, intervals excluding zero). This
  matters because global SHAP summary plots — the standard figure in this
  literature — are built over exactly that population.
- **Rank correlation can hide total disagreement.** On IEEE-CIS, logistic
  regression and the random forest agree at ρ = 0.553, which sounds substantial,
  but share almost no top-five features (Jaccard 0.013).
- **A meaningful feature space is not enough.** IEEE-CIS has 23 interpretable
  features out of 390, yet all three models rank an opaque feature first for 97.5%
  to 99.8% of alerts.
- **Two standard explainers disagree about the same prediction**, and worse as the
  model gets more complex: ρ falls from 0.651 to 0.309 on ULB.

The methods and results chapters are drafted. Every table and figure is generated
from the results files rather than typed in, so a reported number cannot disagree
with the experiment that produced it.

## What I plan to do next

1. **Recruit a second rater** — November. This is the only part of the work blocked
   on something other than computation, and it has been open since mid-October.
2. **Repeat the IEEE-CIS experiments at full scale** — December. They currently use
   a 300,000-row chronological subsample because of memory limits, so conclusions
   there strictly concern the first 85 days.
3. **Add confidence intervals to the budget-level rates** — December. This would
   establish whether the precision/recall divergence in point 2 is real or within
   noise.
4. **Extend the imbalance experiment to a second architecture** — January, to test
   whether the interaction holds beyond gradient boosting.
5. **Write the results and discussion chapters**, and draft a short practitioner
   checklist for reporting explanation quality.

**One request.** Objective 4 needs a second independent rater to score the
explanation reports against the rubric, so that I can report inter-rater agreement.
It does not need machine-learning expertise — the rubric asks whether a named
feature, a direction and a sensible next step can be read off an explanation — and
it is roughly two to three hours of work on 100 alerts per dataset. If you could
rate a sample yourself, or suggest someone, that would unblock the last objective.

I would also welcome your view on whether to prioritise the full-scale IEEE-CIS rerun
or the second architecture, if there is not time for both.

Thank you again for the review. The sample-size problem in point 3 would have gone
unnoticed without it.

Best regards,
Namig Planov
