# Draft email to Prof. Abdul Razaque

*Written 21 November 2026, after all five points of his second review were
completed. Edit before sending — particularly the closing two questions, which
assume you want his steer on prioritisation and his help finding a rater.*

---

**Subject:** Thesis update — all five review points completed, with results

Dear Professor Razaque,

I have completed the five points from your last review. Below is what each one
produced, followed by two questions.

**1. Why the two datasets behave so differently.** The contrast turned out to
be mostly arithmetic. At a fixed alert budget, precision and recall are both
capped by the ratio of review capacity to fraud count, and because the budget
fixes the number of alerts, dividing either rate by its ceiling gives the same
quantity. ULB reaches 83–86% of what is attainable and IEEE-CIS 91–98%, so
once normalised the IEEE-CIS models are the stronger ones — the opposite of
what the raw rates suggest. Two genuine differences remain underneath. Fraud is
far more separable on ULB: 86–89% of its frauds score above the 99th percentile
of legitimate traffic, against 32–47% on IEEE-CIS. And IEEE-CIS is worse
calibrated (expected calibration error 0.09–0.34 against 0.0002–0.056), though
ULB's very low base rate partly flatters that comparison, so I do not claim ULB
is straightforwardly better calibrated.

**2. What the alert budgets mean in practice.** I swept eleven budgets to find
where review capacity first matches the fraud present, which is the point where
the binding constraint switches from recall to precision. ULB crosses at 0.20%
and IEEE-CIS not until 5.00%, so the appropriate budget differs between them by
roughly a factor of twenty-five and is set by the base rate rather than by model
quality. This also showed that the 0.5% budget I had been reporting is already
past ULB's crossing: at 0.1% the same model reaches precision 0.906 and recall
0.713. Against trivial baselines at the same budget, the models catch 172 times
more fraud than random ranking on ULB and 24 times more on IEEE-CIS, while
ranking by transaction amount alone is close to useless on both.

**3. Why resampling affects explanations more than random initialisation.** I
repeated the factorial design on both datasets and with both explainers, giving
four independent tests, and the ordering holds in all four: changing the
resampling regime moves the explanation further than changing the seed or the
model configuration. The mechanism, though, requires a qualification on what I
reported previously. The models' own scores also diverge under resampling
(rank agreement 0.68, against 0.90 under a seed change), so this is
substantially a real change in the decision function rather than an explainer
artefact — resampling changes the model in ways AUC-PR does not register. I
also tested and rejected the off-manifold explanation: SMOTE's synthetic points
sit closer to real fraud than real frauds sit to one another. One caveat worth
stating plainly: on IEEE-CIS with LIME the absolute agreement is near the floor
in every comparison, so that fourth replication supports the direction of the
result but carries little of its weight.

**4. Statistical interpretation.** Every table now leads with Cohen's *d* and
reports how many of the five top-ranked features actually change identity,
which is the figure with an operational meaning. The *p*-values are reported
last, with the caveat that at n = 400 they are all around 10⁻⁶⁷ and say only
that the sample is large.

**5. Reproducibility.** I have written a reproducibility document covering the
environment and package versions, the recorded experimental settings, the
resampling procedures, the evaluation protocol, the order in which the scripts
reproduce the results, the limitations, and a table setting out how this
methodology differs from the reviewed literature.

The Thesis I progress report is complete and covers all of the above. Every
table and figure in it is generated from the results files rather than typed
in, so no reported number can disagree with the experiment that produced it.

**Two questions.**

First, on priorities. Three items remain: bootstrap confidence intervals on the
budget-level precision and recall rates, repeating the IEEE-CIS experiments at
full scale to remove the subsample limitation, and extending the imbalance
experiment to a second architecture. My own inclination is to do the second
architecture first, because generalisation beyond gradient boosting is the one
claim the stability finding still cannot make, and to add the bootstrap
intervals alongside it since they are quick. Would you agree, or would you
rather see the full-scale rerun first?

Second, Objective 4 still needs a second independent rater so that I can report
inter-rater agreement. It requires no machine-learning background — the rubric
asks whether a named feature, a direction of effect and a sensible
investigative next step can be read off an explanation — and it is roughly two
to three hours of work. If you could rate a sample yourself, or suggest
someone, that would close the last objective.

Thank you for the review. The qualification in point 3 would not have surfaced
without your asking for the mechanism rather than the comparison.

Best regards,
Namig Planov
