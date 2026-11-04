# Chapter 3: Methodology

## 3.1 Research design

The study follows a mixed-methods design with a dominant quantitative strand, as set out
in the research proposal. Objectives 1–3 are quantitative and experimental: measurable
properties of explanation outputs are computed under controlled manipulation of a single
variable, and differences are assessed statistically. Objective 4 adds a small structured
qualitative component, because actionability cannot be reduced to a scalar and must be
judged against a rubric.

The unit of analysis requires emphasis, because it distinguishes this work from most of
the literature reviewed in Chapter 2. The objects being measured are not transactions and
not models but **explanation outputs**: the attribution vectors that SHAP and LIME produce
for individual predictions. Transactions and models are instruments for generating those
vectors. A study that reports a model's AUC-PR and then displays a SHAP summary plot has
measured the instrument and not the object.

All experiments are implemented in Python 3.11 and version-controlled. Every number
reported in Chapter 4 is written to a JSON file by the script that computed it, and the
summary tables and figures are generated from those files rather than transcribed, so a
reported value cannot drift from the experiment that produced it.

## 3.2 Data and preparation

### 3.2.1 The two datasets

**ULB Credit Card Fraud Detection** contains 284,807 transactions by European cardholders
over 48 hours in September 2013, of which 492 (0.172%) are fraudulent. Of its 30 features,
28 are principal components named V1–V28; only *Time* and *Amount* are untransformed. It
is included because it is the field's dominant benchmark, which makes results comparable
with prior work, and because its anonymised features supply the negative case for
Objective 4.

**IEEE-CIS Fraud Detection**, provided by Vesta Corporation, contains 590,540 labelled
e-commerce transactions. The competition's test partition is excluded throughout, because
its labels are withheld; all 590,540 labelled rows come from the training partition. Of
these, a **chronological prefix of 300,000 rows** is used for the baseline and
explanation-quality experiments and 150,000 for the imbalance experiment, in both cases
because explaining 1,000 instances against a 390-feature forest exceeded available memory
at full size. The subsample is a prefix rather than a random draw so that the time ordering
is preserved. The 300,000-row prefix spans 85 days and carries a 3.37% fraud rate; the full
file spans 182 days.

Columns more than 90% empty are dropped, which removes two of the 392 candidate features
and leaves **390**. Remaining missing values are handled per model, as described in §3.3.

Both datasets are public and already de-identified at source. Neither is redistributed:
the ULB data is licensed under the Open Database License with its contents under the
Database Contents License, and the IEEE-CIS data under the terms of the Kaggle competition.

### 3.2.2 Time-ordered splits

Every split in this study is **chronological**, not random. Transactions are sorted by
their timestamp — *Time* for ULB, *TransactionDT* for IEEE-CIS — and the first 70% become
the training set, the last 30% the test set. ULB therefore trains on 199,364 transactions
and is tested on 85,443 containing 108 frauds; the IEEE-CIS prefix trains on 210,000 and
is tested on 90,000 containing 3,759 frauds.

The justification is that fraud is not stationary. Measured across chronological deciles,
the fraud rate varies several-fold within both datasets (Figure 1). A random split lets a
model learn from transactions that occur *after* those it is evaluated on, which inflates
every reported metric and does not correspond to any deployment. The cost of the
chronological choice is that reported performance is lower than the published figures most
studies quote for these datasets; that gap is a property of the evaluation protocol, not of
the models, and is discussed in §4.1.

Feature scaling is fitted on the training partition only and applied to the test partition,
so that no test-set statistic informs training.

## 3.3 Models

Three classifiers span the interpretability–performance range the literature contrasts:

| Model | ULB configuration | IEEE-CIS configuration |
|---|---|---|
| Logistic regression | `max_iter=3000`, `class_weight="balanced"` | `max_iter=1000`, `class_weight="balanced"` |
| Random forest | 300 trees, `min_samples_leaf=2`, `class_weight="balanced_subsample"` | 150 trees, `min_samples_leaf=5`, `class_weight="balanced_subsample"` |
| XGBoost | 400 trees, depth 6, learning rate 0.1, subsample 0.9, `colsample_bytree=0.9`, `scale_pos_weight` set to the negative/positive ratio | identical |

The two datasets differ only where size forced it: the IEEE-CIS forest is smaller and its
leaves larger because 390 features over 210,000 rows is substantially more expensive than
30 features over 199,364. All three models address class imbalance through instance
weighting rather than resampling, which keeps the baseline distinct from the resampling
regimes deliberately manipulated in §3.7.

Missing values are handled per model, and the three views of the test matrix are recorded
because the explanation stage must feed each model the representation it was fitted on:
logistic regression receives the standardised matrix, the random forest the
median-imputed matrix (medians computed on training rows only), and XGBoost the raw matrix,
since it consumes missing values natively.

Hyperparameters are fixed a priori at conventional values and are not tuned. This is a
deliberate limitation: the research question concerns the explanations these models produce,
and a tuning search would introduce a selection process whose interaction with explanation
quality is itself unmeasured.

## 3.4 Evaluating detection performance

### 3.4.1 The alert budget

Detection performance is reported at a fixed **alert budget**, which is defined as follows.

> An alert budget *b* is the share of transaction volume that an investigation team has the
> capacity to review. Given a model's scores on the test partition, the alerts it generates
> at budget *b* are the top *b* fraction of transactions ranked by score, so the budget
> fixes the number of cases reviewed at *k = ⌊b · n⌋* and the model's only influence is
> *which* cases fill those *k* slots. Three budgets are reported throughout —
> **b = 0.1%, 0.5% and 1%** — and 0.5% is used wherever a single operating point is
> required.

The construct is operational rather than statistical. A fraud team employs a fixed number
of analysts who can each clear a fixed number of cases per shift, so review capacity is
approximately constant and approximately independent of how much fraud is present. A
threshold on predicted probability does not have this property: the same threshold yields
different alert volumes on different days, and a model whose scores are poorly calibrated
can exhaust a team's capacity without being any better at ranking. Fixing *k* and asking
what fills it reproduces the decision a fraud team actually faces, and makes the three
models directly comparable because each is permitted exactly the same number of
interventions.

At the 0.5% budget this means 427 alerts on the ULB test partition and 450 on IEEE-CIS.

### 3.4.2 Precision and recall at the budget

Two complementary rates are reported at each budget.

**Recall at budget** is the share of all frauds in the test partition that appear among the
*k* alerts — what the team catches. **Precision at budget** is the share of the *k* alerts
that are genuinely fraudulent — what the team's time buys. Both derive from the same
confusion counts, and reporting either alone is misleading in a way the two datasets
demonstrate sharply. At b = 0.5%, logistic regression on ULB achieves recall 0.861 at
precision 0.218, while XGBoost on IEEE-CIS achieves precision 0.984 at recall 0.118. The
first reviews four legitimate transactions for every fraud but catches most fraud; the
second wastes almost no analyst time and misses seven frauds in eight. These are opposite
operational failure modes, and a report of recall alone, or of precision alone, would
describe them as comparable.

Because *k* is fixed, precision and recall at a budget are related by
*precision = recall · F / k*, where *F* is the number of frauds in the test partition. They
are therefore not independent evidence; they are the same count expressed against two
different denominators, one operationally meaningful to the customer and one to the
analyst. Both are reported because practitioners need both, not because they constitute
two tests.

### 3.4.3 Threshold-free measures and their limits

AUC-PR and ROC-AUC are reported alongside the budget-level rates. AUC-PR is the primary
threshold-free measure because, at base rates of 0.172% and 3.37%, ROC-AUC is dominated by
the enormous negative class and compresses the differences that matter. Accuracy is not
reported at all: a classifier that answers "legitimate" for every ULB transaction is
99.83% accurate and catches nothing.

AUC-PR is nonetheless insufficient on its own, and this study's own results show why. On
ULB the random forest attains the best AUC-PR (0.8196) while logistic regression attains
both the best precision and the best recall at the 0.5% budget. A threshold-free average
over all operating points can rank models differently from the single operating point a
team will actually deploy. Both are therefore reported, and where they disagree the
disagreement is stated rather than resolved.

## 3.5 Generating explanations

### 3.5.1 SHAP

SHAP attributions are computed with `TreeExplainer` for the two tree ensembles and
`LinearExplainer` for logistic regression, in both cases the exact method appropriate to
the model class rather than the sampling-based `KernelExplainer`. Each model is explained
on the input representation it was fitted on (§3.3).

Explanations are generated for two populations, because they answer different questions.
The **alert queue** is the top 400 test transactions ranked by a reference model — the
random forest on ULB, XGBoost on IEEE-CIS — and represents what an analyst actually sees.
The **random sample** is 600 further transactions drawn without replacement from the
remainder, and represents the population the model scores. Together these give 1,000
instances per configuration, matching the proposal's commitment of at least 1,000 instances
per metric.

### 3.5.2 LIME and the locality kernel

LIME is applied to the same instances, with continuous features left undiscretised and the
full feature set requested, so that its attribution vector is directly comparable with
SHAP's.

LIME requires one methodological decision that the literature routinely leaves implicit
and that materially affects the result. LIME weights each perturbation by
exp(−*D*²/*w*²), where *D* is a distance in feature space and *w* is a kernel width
defaulting to √*d* · 0.75. On ULB, with *d* = 30, this default produces local surrogates
that fit well: median local *R*² is 0.82–0.96 on the alert queue. On IEEE-CIS, with
*d* = 390, it does not. At the head of the alert queue the model's predicted probability is
approximately 0.99996 and its response surface is locally flat, so a narrow kernel draws
only perturbations whose model output is identical, the weighted regression target has no
variance, and the returned coefficients are numerical noise — magnitudes of order 10⁻²⁹, at
a local *R*² of 0.0002.

Three findings constrain how this is handled. Increasing the sample count does not repair
it: *R*² remains near 0.001 at 10,400 samples. Widening the kernel does: at twice the
default, median *R*² at the head of the queue is 0.99. But the width that works is a
property of the instance and not of the dataset — measured along the alert queue, where
predicted probabilities differ only in the fifth decimal, the best multiplier falls from
2× at positions 0–11 to 1× at positions 200–211, and at the latter positions widening the
kernel *reduces* *R*² from 0.97 to 0.51 by destroying locality.

Consequently the kernel width is selected **per instance**, by maximising the local
surrogate's *R*² over multipliers {1, 2, 3, 4} of the default. This criterion depends only
on how well LIME's own surrogate fits the model it is approximating; it never consults the
SHAP vector against which the result is later compared, so it cannot inflate the agreement
it feeds. Instances whose best surrogate still explains less than half the local variance
(*R*² < 0.5) are **excluded and counted**, because a rank correlation computed against a
surrogate that explains none of the local behaviour measures nothing about the model.

The condition held constant across the two datasets is therefore not the literal kernel
formula but the substantive requirement that LIME's surrogate actually fit. Copying the
parameter would have matched the two experiments in appearance while leaving them
incomparable in fact. Sample count is 2,000 on IEEE-CIS against ULB's 800, since fitting
390 coefficients from 800 draws is a 2:1 ratio where ULB enjoys 27:1; matching the ratio
exactly would require 10,400 draws for no measured gain in *R*².

## 3.6 Measuring explanation quality

Four families of measure are implemented in a single module so that they are computed
identically across every model, dataset and explainer. Each takes an attribution vector
and returns a scalar.

### 3.6.1 Fidelity

Fidelity asks whether an attribution tracks the model's actual behaviour. It is measured by
**deletion**: the features an attribution ranks most important are replaced one at a time
with their training-set mean, and the model's predicted probability is recorded after each
replacement. A faithful attribution should name features whose removal moves the
prediction. The score is the absolute fall from the initial prediction to the mean of the
curve, over the top *k* = 20 features, which is held identical across datasets so that the
numbers compare.

Fidelity is measured on both populations of §3.5.1, because they behave differently and the
difference is itself a result (§4.3). A measure reported on one population only is not
comparable with one reported on the other.

Two implementation choices are recorded because they changed the results. First, the drop
is reported as an **absolute** change in probability rather than a ratio: dividing by the
starting prediction looks tidier but explodes on low-scored transactions, where a ULB model
outputs approximately 10⁻⁶ and any movement becomes a ratio in the thousands. Second, the
curve is truncated to *k* = 20 rather than walking all *d* features, because an insertion
curve that fills only 20 of 390 features never approaches the true prediction, which drives
an insertion-minus-deletion measure negative for reasons unrelated to explanation quality.
Deletion fidelity was adopted in place of that measure for exactly this reason.

### 3.6.2 Stability, in two distinct senses

The proposal asked for stability, and the supervisor's October review asked for stability
"across multiple runs". These are different questions, and this study measures both
separately because a method can pass one and fail the other.

**Input stability** is max-sensitivity: the largest relative change in the attribution
vector when the input is perturbed within an ε-ball, with ε = 0.05 relative to each
feature's magnitude and 8 perturbations per instance. The model is fixed; the input moves.
It asks whether an explanation would survive a slightly different transaction. Lower is
better, and it is computed over 1,000 instances per model.

**Training stability** is agreement across retraining seeds: one architecture and one
dataset are used to fit ten models differing only in random seed, and the SHAP rankings
each produces for the *same* fixed alert queue are compared against a reference seed. The
input is fixed; the model moves. It asks whether an explanation is a property of the data
or an artefact of one training run.

Reporting only the first would leave the supervisor's question unanswered; reporting only
the second would miss the instability that input perturbation exposes. The two also serve
different arguments: training stability functions as the control for the imbalance
experiment (§3.7), since a finding that resampling changes explanations is only meaningful
if ordinary retraining does not.

### 3.6.3 Agreement

Two attribution vectors are compared in two ways, because they capture different things.
**Spearman rank correlation** on absolute attributions compares the full ordering of all
*d* features. **Jaccard overlap at *k* = 5** compares only the set of five features an
investigator would actually read.

These can diverge sharply, and the divergence is itself a result. Where the two measures
disagree — a high rank correlation alongside a near-zero top-5 overlap — the correct
reading is that two explanations order the long tail of unimportant features similarly
while disagreeing about everything an analyst would act on. Rank correlation alone would
report such a pair as broadly in agreement. Both are therefore always reported together.

Agreement is computed in two directions: **cross-explainer** (SHAP against LIME on one
model and one instance) and **cross-model** (SHAP on two different models for the same
instance). Rank correlation is undefined when either vector is constant; such instances are
excluded from the aggregate and their number is recorded.

### 3.6.4 Semantic actionability

Objective 4 asks whether an explanation can be acted on. The quantitative half is measured
on IEEE-CIS, which has named features, by classifying each feature as **opaque** or
**semantic**. Opaque features are Vesta's engineered V-columns, the masked `id_*` columns,
and the `C*` and `D*` counting and time-delta families, all of which are uninterpretable in
the way ULB's principal components are. Everything else — transaction amount, product code,
card and address fields, email domains — carries a meaning an investigator could pursue. Of
the 390 features retained, 23 are semantic.

Two rates are reported per model: the share of alerts whose single highest-attributed
feature is semantic, and the mean share of the top five that are semantic. ULB scores zero
on both by construction, which is the comparison Objective 4 exists to make.

The qualitative half scores explanation reports against a predefined rubric covering
whether a named feature, a direction of effect and a plausible investigative next step can
be derived. It requires two independent raters with agreement reported, and is outstanding
at the time of writing.

## 3.7 The imbalance experiment

Objective 3 tests whether class-imbalance handling distorts explanations. The design holds
everything constant except the resampling regime: one architecture (XGBoost at the
configuration in §3.3), one chronological split, one test partition, and three training
regimes — **natural imbalance**, **random undersampling** of the majority class, and
**SMOTE** oversampling of the minority class with five neighbours.

Each regime produces a model; each model explains the same fixed alert queue; and the
resulting attributions are compared against the natural-imbalance regime's. Predictive
performance is recorded for every regime so that it can be held as a covariate, which is
the crux of the design: if detection quality is materially unchanged while explanations
diverge, the divergence cannot be dismissed as the by-product of a worse model.

The null hypothesis is that explanation rankings are invariant to the resampling regime.
The experiment is run independently on both datasets.

## 3.8 Statistical inference

Every point estimate is reported with a **95% percentile bootstrap confidence interval**
over 2,000 resamples of the per-instance values. Intervals rather than point estimates are
reported throughout because several of this study's conclusions turn on whether two
intervals overlap, and because a sample of 80 instances and a sample of 1,000 can yield
similar means with very different precision — a difference that changed a conclusion in
this work and is described in §4.2.

Paired comparisons use the **Wilcoxon signed-rank test**, a non-parametric paired test
appropriate to per-instance agreement scores that are bounded and not normally distributed.

A caveat is stated here because it governs how Chapter 4 should be read. With 400 paired
instances, the Wilcoxon test is powered to detect differences far smaller than any that
would matter operationally, and every cross-model comparison in this study returns
*p* ≈ 10⁻⁶⁷. Such a *p*-value carries no information beyond "n is large". **Cohen's *d*** is
therefore reported alongside every test and carries the substantive argument; the *p*-value
is reported for completeness and because the proposal committed to it. Where the two might
conflict, the effect size governs the interpretation.

## 3.9 Reproducibility and validation of the apparatus

All randomness is seeded. Experiments are driven by numbered scripts that each write a JSON
file of results, and the summary tables and figures are generated from those files, so no
reported number is retyped. The seed, the number of instances per metric, the fidelity
curve length and the top-*k* value are written into each results file alongside the
measurements, so that two runs can be compared for condition drift rather than assumed
comparable — an omission that concealed a genuine inconsistency during this work, described
in §4.2.

The metric implementations are themselves validated before use, on synthetic models whose
true feature importances are known by construction. This control is a necessary precaution:
a measure that cannot recover a known answer cannot be trusted on an unknown one. The
validation checks are reported in §4.1.

## 3.10 Limitations of the measurement approach

The principal limitation is deliberate. The strongest missing evidence in this field
concerns whether explanations help **professional fraud analysts**, and recruiting such
analysts is not feasible within a master's project. This study therefore measures proxy
properties — fidelity, stability, agreement and rubric-scored actionability. These are
*necessary but not sufficient* conditions: an explanation that is unfaithful or unstable
cannot be useful, but one that is faithful and stable is not thereby proven useful. The
findings constrain the field's claims rather than settling them.

Three further limitations follow from the design. The IEEE-CIS experiments use a
chronological subsample of 300,000 of 590,540 labelled rows, so conclusions about that
dataset strictly concern its first 85 days. Hyperparameters are untuned, so the models are
reasonable rather than optimal. And the imbalance experiment manipulates one architecture;
whether the interaction it finds generalises across model classes is not established here.
