# Progress Report 3: Research Proposal

## 1. Problem Statement

### 1.1 Introduction to the problem

Machine learning models that detect fraudulent payment transactions are accurate but
opaque. Gradient-boosted ensembles and deep architectures outperform interpretable
alternatives on every published benchmark, yet they cannot state why a particular
transaction was flagged. Because a flag leads to a declined payment, a frozen account or
a report to a financial intelligence unit, the industry has adopted post-hoc explainable
AI (XAI) — principally SHAP and LIME — as the standard remedy.

The literature review conducted in Progress Report 2 found that this remedy is largely
unverified. Three findings drive the present proposal. First, explanation quality is
asserted rather than measured: a systematic review of 49 studies reports that over 80%
use the model's predictive performance as a proxy for the quality of its explanations,
although the two properties are logically independent. Second, where explanations have
been tested on real fraud analysts, the results have been negative — Weerts et al. (2019)
found no significant improvement in alert-processing performance when SHAP was available,
and Jesus et al. (2021) concluded that popular explainers had "a worse impact than
desired". Third, the field's dominant benchmark, the ULB European card dataset, contains
features V1–V28 that are anonymised principal components, so a SHAP attribution on that
data states that "V14" drove the decision — a statement no investigator can act on and no
customer can be told.

The problem this research addresses is therefore not that fraud models lack explanations.
It is that **the explanations already being produced and relied upon have not been shown
to be faithful, stable or actionable**, and that a methodological interaction specific to
fraud data — the use of resampling to handle extreme class imbalance — may distort them in
a way the field has not tested.

### 1.2 Context and relevance

The context is both regulatory and operational. The EU AI Act classifies
financial-services and creditworthiness systems as high-risk, and Article 22 of the GDPR
continues to constrain solely automated decision-making, so institutions must document
how automated decisions are reached. Explanations are consequently being generated at
scale to satisfy obligations whose evidential basis has not been examined.

Operationally, fraud alerts are triaged by human analysts under time pressure, and false
positives impose direct costs on customers. If an explanation does not improve an
analyst's judgement, it consumes review time without benefit; worse, if it is unfaithful
to the model it may actively mislead.

The real-world implications are therefore concrete. A bank that documents an automated
decision with an unfaithful explanation satisfies a regulator with an artefact that does
not describe what the model did; an analyst who trusts an unstable attribution may clear
a fraudulent transaction or escalate a legitimate one; and a customer denied a payment
receives a reason that is, in the case of the field's dominant benchmark, not a reason at
all. Each of these failures is invisible unless explanation quality is measured directly.

For computer science and data analytics the relevance is methodological. Explanation
methods are increasingly treated as a validated component of a machine-learning pipeline
despite being evaluated far more loosely than the models they describe. Establishing
whether standard practice survives direct measurement is a contribution to the validation
of machine-learning systems generally, and the imbalance–explanation interaction is a
concrete, testable instance of that broader question.

---

## 2. Research Objectives

### 2.1 Primary objective

To evaluate empirically the quality of post-hoc explanations produced for transactional
fraud-detection models — specifically their **fidelity** to the underlying model, their
**stability** under small input perturbations, and their **semantic actionability** — and
to determine whether class-imbalance handling systematically distorts them.

### 2.2 Secondary objectives

1. **Establish a reproducible modelling baseline.** Train logistic regression, random
   forest and XGBoost classifiers on two public fraud datasets using time-ordered rather
   than random splits. *Measurable outcome:* AUC-PR and recall at fixed alert budgets
   reported for each model–dataset pair, within the range of published results, with code
   and fixed random seeds released.

2. **Implement and apply direct explanation-quality metrics.** Compute fidelity
   (deletion–insertion curves and infidelity), stability (maximum sensitivity under
   ε-bounded perturbation) and cross-method agreement (Spearman rank correlation and
   Jaccard overlap of top-*k* features between SHAP and LIME). *Measurable outcome:*
   each metric computed over a sample of at least 1,000 test instances per configuration
   with bootstrap confidence intervals.

3. **Test the imbalance–explanation interaction under controlled conditions.** Train
   identical architectures under three regimes — natural imbalance, random undersampling
   and SMOTE — and compare explanations across them while holding predictive performance
   as a covariate. *Measurable outcome:* a statistical test of the null hypothesis that
   explanation rankings are invariant to resampling regime, with effect sizes reported.

4. **Compare semantic actionability across feature representations.** Generate
   explanation reports for matched alerts from the PCA-anonymised ULB dataset and the
   semantically named IEEE-CIS dataset, and score them against a predefined rubric
   covering whether a named feature, a direction and a plausible investigative next step
   can be derived. *Measurable outcome:* rubric scores for at least 100 alerts per
   dataset, rated independently by two raters with agreement reported.

5. **Release reusable artefacts.** Publish the metric implementations, experimental
   configuration and a short practitioner checklist for reporting explanation quality.
   *Measurable outcome:* a public repository that reproduces every reported figure from
   raw data.

Objectives 2, 3 and 4 operationalise research questions RQ2, RQ3 and RQ4 of Progress
Report 2 and address Gaps 1, 2 and 4 identified there.

---

## 3. Selected Methodology

### 3.1 Research approach

The study follows a **mixed-methods design with a dominant quantitative strand**.
Objectives 1–3 are quantitative and experimental: measurable properties of explanation
outputs are computed under controlled manipulation of a single variable (the resampling
regime), and differences are assessed statistically. Objective 4 adds a small structured
qualitative component, since actionability cannot be reduced to a scalar and must be
judged against a rubric.

This approach is appropriate because the identified gap is precisely that the field
substitutes narrative claims for measurement. A purely qualitative study would reproduce
that weakness; a purely quantitative one could not address whether an explanation is
*meaningful*, which is the substance of Gap 4. Computational experimentation is also the
only design that permits the imbalance manipulation required by Objective 3, because the
counterfactual — the same model trained under a different regime — does not exist in
observational data.

### 3.2 Data collection methods

The research uses **secondary data**: two public benchmark datasets, with no new data
collected from individuals.

**ULB Credit Card Fraud Detection** contains 284,807 transactions made by European
cardholders over two days in September 2013, of which 492 (0.172%) are fraudulent.
Features V1–V28 are principal components; only *Time* and *Amount* are untransformed. It
is included because it is the field's dominant benchmark, which makes results directly
comparable with prior work, and because its anonymised features provide the negative case
for Objective 4.

**IEEE-CIS Fraud Detection**, provided by Vesta Corporation, contains 590,540 labelled
e-commerce transactions with approximately 3.5% fraud and several hundred features, many
semantically named — transaction amount, product code, card and address fields, email
domains, counting features and time deltas. It supplies the interpretable feature space
Objective 4 requires, and its time-ordered train/test separation supports the use of
temporal rather than random splits.

The primary objects of measurement are not the transactions themselves but the
**explanation outputs generated from them**, which are produced experimentally rather
than collected. Public benchmarks are appropriate because they permit exact
reproduction, avoid processing personal data, and allow comparison against the published
results the review criticised.

### 3.3 Data analysis techniques

Models will be implemented in Python 3.11 using scikit-learn for logistic regression and
random forests, XGBoost for gradient boosting, and imbalanced-learn for SMOTE and random
undersampling. Explanations will be generated with the `shap` library (TreeExplainer for
tree ensembles, KernelExplainer where model-agnostic comparison is required) and `lime`.

Predictive performance will be reported as AUC-PR and recall at fixed alert budgets
rather than accuracy, which is uninformative at a 0.17% base rate. Explanation quality
will be analysed through deletion–insertion curves, infidelity and maximum-sensitivity
scores, and rank-agreement statistics between explainers.

Inference will use paired non-parametric tests across instances — Wilcoxon signed-rank
for paired comparisons between resampling regimes — with bootstrap confidence intervals
and Holm correction for multiple comparisons. Inter-rater agreement on the actionability
rubric will be reported using Krippendorff's alpha. All experiments will use fixed seeds
and be version-controlled in Git.

---

## 4. Proposed Research Design

### 4.1 Research structure

The work proceeds in seven stages. **(S1)** Environment setup, data acquisition and
exploratory analysis. **(S2)** Baseline model development with time-ordered splits and
performance benchmarking against published figures. **(S3)** Construction of the
explanation-generation pipeline for SHAP and LIME. **(S4)** Implementation of the
explanation-quality metrics and validation of those implementations on synthetic models
whose true feature importances are known by construction — a necessary control, since a
metric that cannot recover a known answer cannot be trusted on an unknown one.
**(S5)** The controlled imbalance experiment. **(S6)** The semantic actionability study.
**(S7)** Statistical analysis, error analysis and writing.

### 4.2 Timeline

| Period | Activity |
|---|---|
| October 2026 | Finalise literature review; S1 — environment, data acquisition, exploratory analysis |
| November 2026 | S2 — baseline models, time-ordered splits, performance benchmarking |
| December 2026 | S3 — explanation-generation pipeline; interim supervisor review |
| January 2027 | S4 — metric implementation and validation on synthetic controls |
| February 2027 | S5 — controlled imbalance experiment |
| March 2027 | S6 — semantic actionability study and rubric rating |
| April 2027 | S7 — statistical analysis, error analysis, results chapters |
| May 2027 | Full draft, revision following supervisor feedback |
| June 2027 | Final submission and defence |

### 4.3 Limitations

The most significant limitation is deliberate and should be stated plainly. The review
found that the strongest missing evidence concerns whether explanations help
**professional fraud analysts**, yet recruiting such analysts is not feasible within a
master's project. This study therefore measures proxy properties — fidelity, stability,
agreement and rubric-scored actionability. These are *necessary but not sufficient*
conditions for a useful explanation: an explanation that is unfaithful or unstable cannot
be useful, but one that is faithful and stable is not thereby proven useful. The findings
constrain the field's claims rather than settling them.

Further limitations follow from the data. The ULB dataset covers two days in 2013 and is
not representative of current fraud; IEEE-CIS is specific to e-commerce. Neither is a live
production stream, so results concerning temporal drift are indicative only. The study
examines post-hoc explainers applied to opaque models; inherently interpretable
alternatives (Rudin, 2019) enter only as a comparison point. Finally, KernelSHAP is
computationally expensive, so model-agnostic analyses will use sampled instances with
reported confidence intervals rather than exhaustive computation.

### 4.4 Ethical considerations

No human participants will be recruited and no personal data will be collected. Both
datasets are public and already de-identified at source — ULB through PCA transformation,
IEEE-CIS through anonymisation by the data provider — and no attempt at re-identification
will be made. Licence terms will be observed: the ULB data is distributed under the Open
Database License with its contents under the Database Contents License, and the IEEE-CIS
data under the terms of the Kaggle competition; neither will be redistributed.

The rubric-based assessment in Objective 4 involves the author and one additional rater
evaluating machine-generated explanations rather than human subjects, so it does not
constitute human-subjects research; ADA University's ethics guidance will nonetheless be
consulted and formal approval sought if the institution deems it necessary.

One dual-use consideration applies. Demonstrating that resampling distorts explanations,
or that explainers are unstable under perturbation, is information that could in
principle assist an adversary seeking to evade detection. Reporting will therefore
emphasise measurement and mitigation rather than exploitation, consistent with the
responsible framing adopted by Slack et al. (2020). Finally, releasing code and seeds is
treated here as an ethical obligation as much as a methodological one, given that the
review's central criticism is of claims that cannot be checked.

---

**Word count (Sections 1–4, excluding headings, references and this line): 1,920 words.**

---

## References

Jesus, S., Belém, C., Balayan, V., Bento, J., Saleiro, P., Bizarro, P., & Gama, J. (2021). How can I choose an explainer? An application-grounded evaluation of post-hoc explanations. *Proceedings of FAccT 2021*, 805–815.

Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206–215.

Slack, D., Hilgard, S., Jia, E., Singh, S., & Lakkaraju, H. (2020). Fooling LIME and SHAP: Adversarial attacks on post hoc explanation methods. *Proceedings of AIES 2020*, 180–186.

Weerts, H. J. P., van Ipenburg, W., & Pechenizkiy, M. (2019). A human-grounded evaluation of SHAP for alert processing. *arXiv:1907.03324*.
