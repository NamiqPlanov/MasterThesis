# Draft reply to Prof. Abdul Razaque

*Plain-language draft. Edit freely before sending — especially the closing ask, which
depends on whether you want to approach raters yourself first.*

---

**Subject:** Response to your review — all five points addressed

Dear Professor Razaque,

Thank you for the review. I have worked through all five points. Here is what changed and
what I found.

**1. The alert budget is now defined.**
An alert budget *b* is the share of transactions the team has capacity to review. The
alerts are the top *b* fraction ranked by model score, so the budget fixes how many cases
are reviewed and the model only decides which cases fill those slots. I report three
budgets — 0.1%, 0.5% and 1% — and use 0.5% where a single operating point is needed. At
0.5% that is 427 alerts on ULB and 450 on IEEE-CIS. The definition is written into the
methods chapter with the reasoning for why a fixed capacity is more realistic than a fixed
probability threshold.

**2. Precision and recall are now reported.**
Both were recoverable from counts I had already stored, so no retraining was needed. They
show something AUC-PR had hidden: the two datasets fail in opposite ways at the same
budget. On ULB, logistic regression catches 86% of fraud but is wrong about four alerts in
five. On IEEE-CIS, XGBoost is right about 98% of its alerts but misses seven frauds in
eight. Reporting only one of the two rates would have made these look comparable.

A second thing surfaced: on ULB the random forest has the best AUC-PR, but logistic
regression wins both precision and recall at the 0.5% budget. The margins are small, so I
present it as a demonstration that the two kinds of measure can disagree, not as proof that
one model is better.

**3. Experimental conditions are now recorded and consistent.**
Every script now writes its conditions — seed, number of instances, curve length, top-*k* —
into its results file, so a mismatch cannot stay hidden. That immediately exposed one: the
stability metric was using 80 instances where the proposal commits to at least 1,000.
Fixing it changed a conclusion, which I describe in point 4.

Making the SHAP-versus-LIME comparison consistent across the two datasets turned out to be
the hardest part, and it produced a finding of its own. LIME weights its samples by
distance, with a default width based on the number of features. On ULB (30 features) this
works well. On IEEE-CIS (390 features) it fails completely: at the top of the alert queue
the model's output is so flat that LIME's samples all return the same prediction, its
internal regression has nothing to fit, and the weights it returns are numerical noise —
its own goodness-of-fit is 0.0002. Using more samples does not help. I now choose the width
per instance by how well LIME's local model fits, and discard any instance where it still
does not fit. That lifts the fit from 0.006 to between 0.83 and 0.998, and all 120
instances now pass. So the condition I hold constant across datasets is that LIME's local
model must actually fit — not that both runs use the same formula, which would have made
the two experiments look matched while leaving them incomparable.

**4. Stability across multiple runs is now measured.**
I treated this as two separate questions, because they fail for different reasons:

- Perturbing the *input* with the model fixed. This is where the sample-size fix mattered.
  At 80 instances the two tree models on ULB looked equally unstable; at 1,000 their
  confidence intervals no longer overlap and XGBoost is twice as unstable as the random
  forest. On IEEE-CIS, XGBoost scores 0.960, meaning a 5% nudge to the input changes the
  explanation by almost as much as the explanation's own size — and XGBoost is the best
  detector on that dataset.
- Perturbing the *seed* with the data fixed. I trained ten XGBoost models differing only in
  random seed and compared their explanations of the same alerts. Agreement is high
  (average 0.873) and AUC-PR varies by only 0.013.

The second one also gave me the control I needed for the imbalance experiment. Retraining
moves the explanation to 0.873; changing the resampling method moves it to 0.45–0.58. So
the divergence I reported earlier is caused by the resampling strategy, not by random
training noise. That was the obvious objection to that result and it is now answered.

**5. Statistical evidence is now attached to every comparison.**
Wilcoxon signed-rank tests and Cohen's *d* on all paired comparisons, with 95% bootstrap
confidence intervals on every point estimate. One honest caveat: with 400 paired instances
every *p*-value comes out around 10⁻⁶⁷, which only tells us the sample is large. I report
the *p*-values because the proposal promised them, but I lead with the effect sizes, which
range from 2.9 to 12.4 and carry the actual argument.

**Where the work stands.** The methods and results chapters are drafted. Everything is in
the repository, and every table and figure is generated from the results files rather than
typed in, so a number cannot disagree with the experiment that produced it.

**One thing I still need help with.** Objective 4 needs a second independent rater to score
the explanation reports against the rubric, so that I can report inter-rater agreement. It
does not need machine-learning expertise — the rubric asks whether a named feature, a
direction and a sensible next step can be read off an explanation — and it is roughly two
to three hours of work on 100 alerts per dataset. If you could rate a sample yourself, or
suggest someone, that would unblock the last objective. It has been the only open item
since mid-October.

Thank you again for the review. The sample-size problem in point 3 would have gone
unnoticed without it.

Best regards,
Namig Planov
