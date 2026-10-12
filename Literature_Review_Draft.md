# Draft Literature Review

**Thesis title:** Explainable AI for Fraud Detection in Financial Transactions
**Student:** [Your Full Name]  ·  **Student ID:** [Your ID]
**Supervisor:** Abdul Razaque
**Programme:** MSc [Computer Science / Data Analytics]
**Date:** 3 October 2026

---

## 1. Introduction to the Literature Review

### 1.1 Purpose and scope

Machine learning now underpins fraud detection in virtually every payment network, yet
the models that perform best — gradient-boosted ensembles and deep neural networks — are
opaque. Their decisions carry direct consequences: a declined transaction, a frozen
account, a customer reported to a financial intelligence unit. Both regulators and
institutions therefore demand that these decisions be explainable, and explainable
artificial intelligence (XAI) has become the dominant response.

The purpose of this review is to establish what is actually known about applying XAI to
transactional fraud detection, and to separate claims that are empirically supported from
those that are merely assumed. The scope is deliberately narrow: supervised machine
learning applied to *transaction-level* fraud (card-not-present fraud, suspicious bank
transfers, and related payment fraud), together with the post-hoc explanation methods
applied to those models. Credit-scoring, insurance fraud and anti-money-laundering (AML)
typologies are drawn on only where their findings transfer.

Three concepts recur and are used here with fixed meanings. *Interpretability* is a
property of a model whose mechanism can be inspected directly. *Explainability* is the
production of a post-hoc account of a model that is not itself interpretable.
*Explanation quality* is the degree to which such an account is faithful to the model and
useful to the person receiving it — a distinct property from the model's predictive
accuracy, and, as this review argues, a far less frequently measured one.

### 1.2 Research questions

The review is guided by four questions:

- **RQ1.** Which XAI techniques dominate transactional fraud detection, and how
  methodologically similar are the studies applying them?
- **RQ2.** How is explanation quality evaluated in this literature, and how often is it
  evaluated at all?
- **RQ3.** What is the interaction between class-imbalance handling — intrinsic to fraud
  data — and the fidelity of post-hoc explanations?
- **RQ4.** Which gaps in the existing evidence can realistically be addressed within a
  master's thesis?

---

## 2. Search Strategy and Sources

### 2.1 Search methodology

Literature was gathered between September and October 2026 from Scopus, IEEE Xplore, the
ACM Digital Library, SpringerLink, ScienceDirect and Google Scholar, supplemented by
arXiv for very recent work. The base query combined a domain term with a method term:

> ("fraud detection" OR "financial fraud" OR "suspicious transaction") AND
> ("explainable AI" OR XAI OR interpretab* OR SHAP OR LIME)

Results were restricted to 2016–2026, 2016 being the publication year of LIME and
therefore the practical start of the modern post-hoc explanation literature. Backward and
forward citation chaining was applied to the most-cited items, which is how the two
human-subject evaluation studies central to Section 3.4 were located; neither appears in
the first page of results for the base query, which is itself indicative of how peripheral
evaluation work remains in this field.

Inclusion required that a study apply an explanation method to a fraud or suspicious-
transaction detection model and report an empirical result. Purely theoretical XAI papers
were included only where foundational (for example the original SHAP and LIME papers).
Excluded were studies without an explainability component, non-English publications, and
vendor white papers.

### 2.2 Sources and a note on source quality

The reviewed corpus is predominantly peer-reviewed: journal articles from *Finance
Research Letters*, *Computers*, *Knowledge and Information Systems*, *International
Journal of Machine Learning and Cybernetics*, *FinTech*, *Artificial Intelligence Review*
and *Nature Machine Intelligence*, and conference papers from NeurIPS, KDD, FAccT and
AIES.

Two caveats are recorded here rather than buried. First, one systematic review used
extensively in Section 3.4 — *Methodological Challenges in Explainable AI for Fraud
Detection* — is at the time of writing a **Research Square preprint and has not completed
peer review**; its findings are reported as preliminary and are, wherever possible,
corroborated against peer-reviewed sources. Second, a frequently circulated title,
"Trustworthy and Interpretable Machine Learning Models for Financial Fraud Detection",
could not be located in any indexed database during this search and has therefore been
excluded rather than cited unverified.

---

## 3. Thematic Analysis of the Literature

### 3.1 Theme 1 — The accuracy–opacity trade-off in fraud models

Fraud detection presents a distinctive statistical problem: extreme class imbalance.
In the European card dataset used across most of this literature, 492 of 284,807
transactions are fraudulent — roughly 0.17%. The field's response has converged on
tree ensembles, chiefly random forests and XGBoost, combined with resampling. SMOTE
(Chawla et al., 2002) remains the default, though Dal Pozzolo et al. (2015) showed that
undersampling distorts posterior probabilities and requires explicit recalibration — a
correction that much applied work still omits.

Baisholan et al. (2025) depart from the norm in a way worth highlighting. Their FraudX AI
framework averages random forest and XGBoost probabilities with threshold optimisation,
reporting 95% recall and 97% AUC-PR, but critically they evaluate **on the dataset's
natural imbalance rather than rebalancing it**. This is methodologically stronger than the
common practice of reporting accuracy on an artificially balanced test set, where metrics
are inflated and unrepresentative of deployment. Their choice also sidesteps the
explanation-distortion problem discussed in Section 3.5, apparently without the authors
framing it that way.

The limitation across this theme is the near-universal reliance on a single dataset,
examined further in Section 3.6.

### 3.2 Theme 2 — Post-hoc explanation as the standard response

Having chosen opaque models, the field explains them after the fact. Two methods dominate.
LIME (Ribeiro et al., 2016) fits an interpretable surrogate in the local neighbourhood of
a prediction. SHAP (Lundberg & Lee, 2017) allocates each feature a contribution derived
from Shapley values, offering consistency guarantees and both local and global views.

What is striking on reading the applied papers together is their methodological
uniformity. Zhou et al. (2023) combine an ensemble with Shapley-value explanations,
producing local explanations for individual alerts and global explanations of model logic,
and frame the work around external stakeholders. The 2026 *Knowledge and Information
Systems* study applies feature importance, SHAP, LIME and DeepLIFT to real-world card
data, finding random forests strongest among classical models and TabTransformer strongest
among deep models. The 2026 *International Journal of Machine Learning and Cybernetics*
study applies SHAP and LIME to suspicious bank transactions, attributing predictions to
transaction amount, risk score, anomaly score and frequency. Baisholan et al. (2025) apply
SHAP to their ensemble.

The pipeline is essentially invariant: *dataset → resampling → tree ensemble →
performance metrics → SHAP → feature-importance plot*. Individually each paper is
competent; collectively they demonstrate replication rather than accumulation. The
explanation stage is consistently treated as a terminal reporting step rather than as a
component with properties of its own that might be measured, compared or falsified. This
answers **RQ1**: the techniques are SHAP and LIME almost to the exclusion of alternatives,
and the studies are more similar than their differing titles suggest.

Comparing these studies on methodological rigour rather than on headline accuracy
separates them more usefully. Baisholan et al. (2025) are strongest on experimental
design: they benchmark against eight baseline models, report AUC-PR rather than accuracy —
the appropriate choice under extreme imbalance, since accuracy is near-99.8% for a model
that predicts "legitimate" unconditionally — and preserve the natural class distribution.
Their weakness is that SHAP enters only as a descriptive postscript, with no analysis of
whether the attributions are stable across runs or consistent between the two ensemble
members being averaged. Zhou et al. (2023) are strongest conceptually, being the only
reviewed study to begin from stakeholder requirements and to distinguish local from global
explanatory needs; but the stakeholder framing is argued rather than tested, as no
stakeholders are asked whether the explanations met their needs. The 2026 *Knowledge and
Information Systems* study is the broadest methodologically, comparing four explanation
techniques including DeepLIFT across both classical and deep architectures, which permits
cross-method comparison that single-explainer studies cannot support; yet agreement
between those four methods — a natural and informative result readily available from such
a design — is not reported. The recurring pattern is that each study's explanatory
component is weaker than its predictive component, and that the comparisons their own
designs would permit go unmade.

### 3.3 Theme 3 — Regulation, governance and trust

A second driver is legal. Wachter et al. (2017) showed that the much-cited GDPR "right to
explanation" is considerably weaker than commonly assumed, yet Article 22's constraints on
solely automated decision-making still shape practice, and the EU AI Act's treatment of
creditworthiness and financial-services systems as high-risk has intensified documentation
requirements.

Gupta et al. (2026) provide the most structured treatment, synthesising 99 Scopus-indexed
articles from 2015–2026 using PRISMA 2020 and SPAR-4-SLR, and organising findings through
an ADO–TCCM meta-framework around trust, governance and transparency. Their contribution
is to position explainability as an organisational and regulatory practice rather than a
purely technical one. Černevičienė and Kabašinskas (2024) reach compatible conclusions
across finance more broadly.

A tension runs through this theme that the literature rarely names. An explanation
produced to satisfy an auditor is a *compliance artefact*: it must be documented,
consistent and defensible. An explanation produced to help an analyst triage an alert in
seconds is a *decision aid*: it must be fast, actionable and cognitively light. These are
different artefacts with different success criteria, yet the applied papers generally
produce one SHAP plot and implicitly offer it to both audiences.

### 3.4 Theme 4 — The evaluation vacuum

This is the most consequential theme, and the one where the field's confidence is least
supported.

Most applied studies validate the *model* and then present explanations without
validating them. The *Methodological Challenges* review (preprint, 2025) quantifies this
across 49 studies from 2021–2025: more than 80% use predictive performance as a proxy for
explanation quality, a substitution the authors call the "evaluation vacuum". Since
predictive accuracy is logically independent of explanation fidelity, it cannot serve as
evidence for it.

Where explanations have been evaluated with human subjects, results have been
discouraging. Weerts et al. (2019) tested whether SHAP helped domain experts assess
alerts and found **no significant difference in alert-processing performance** with
explanations versus without; the model's confidence score remained the dominant cue.
Jesus et al. (2021) went further, running an application-grounded evaluation of LIME,
SHAP and TreeInterpreter with real fraud analysts on a deployed model and real data,
incrementally varying the information shown. They concluded that these popular explainers
have "a worse impact than desired".

These two studies are rarely cited in the applied papers surveyed above. The field's
strongest empirical evidence about whether explanations help the intended user is
negative, and it is largely absent from the work that assumes they do. This is the central
finding of the present review and the direct answer to **RQ2**.

Theoretical work points the same way. Rudin (2019) argues that for high-stakes decisions,
post-hoc explanation of a black box is the wrong strategy, and that inherently
interpretable models should be preferred; fraud detection satisfies every criterion she
sets for "high stakes", yet the interpretable-by-design alternative is barely represented
in this literature. Doshi-Velez and Kim (2017) had already called for rigorous evaluation
taxonomies that remain unadopted here.

### 3.5 Theme 5 — Where imbalance meets explanation

The two previous themes intersect in a way that has only recently been articulated. The
*Methodological Challenges* review names an "explainability–imbalance paradox":
resampling creates synthetic minority instances, and post-hoc explainers subsequently
attribute importance partly to artefacts of that synthetic data rather than to genuine
fraud signal. If correct, this implies the standard pipeline of Section 3.2 may yield
explanations that are faithful to a resampled distribution that does not exist in
production.

This claim is currently supported by a preprint and requires independent verification —
which makes it an attractive target for empirical work rather than a settled result.
Adjacent evidence is consistent with it: Slack et al. (2020) demonstrated that LIME and
SHAP can be deliberately fooled, with their reliance on perturbed samples drawn off the
data manifold as the attack surface. Synthetic oversampling places training points in
precisely such regions. Baisholan et al. (2025), by preserving natural imbalance, avoid
the problem, but do not test for it.

### 3.6 Gaps in the literature

Five gaps follow from the analysis.

**(1) Explanation quality is asserted, not measured.** Fidelity, stability and
plausibility are rarely reported; model accuracy stands in for all three. The two
human-grounded studies that exist found little benefit, and are seldom engaged with.

**(2) The imbalance–explanation interaction is largely untested.** The paradox is
plausible and consequential but rests on one unreviewed synthesis.

**(3) "Explanation" is treated as one thing for one audience.** Analyst, customer and
regulator have incompatible requirements; no reviewed study evaluates one explanation
against more than a single stakeholder's needs, despite Zhou et al. (2023) framing the
problem in stakeholder terms.

**(4) A dataset monoculture undermines the very claim being made.** Most studies use the
ULB European card dataset, whose features V1–V28 are *anonymised principal components*.
A SHAP plot over that dataset reports that "V14" and "V17" drove the decision — statements
with no semantic content. An investigator cannot act on them and a customer cannot be told
them. The field's dominant benchmark is therefore structurally incapable of demonstrating
the practical explanation quality the field claims to deliver. Studies using
semantically meaningful features, such as the 2026 bank-transaction study with its
interpretable amount, frequency and risk-score attributes, are the exception.

**(5) Temporal validity is neglected.** Fraud adapts adversarially, yet almost all
evaluation is on static random splits rather than temporally ordered ones, leaving
explanation stability under concept drift unexamined.

---

## 4. Conclusion and Research Relevance

### 4.1 Summary of key insights

The literature establishes convincingly that ensemble models detect transactional fraud
accurately under severe imbalance, and that SHAP and LIME can be attached to them to
produce plausible-looking attributions. It establishes far less than it appears to about
whether those attributions are faithful, stable, or useful. Methodological convergence has
produced a large body of near-identical pipelines in which the explanation stage is
reported but not examined; the small number of studies that did examine it with real
analysts found minimal benefit; and the field's dominant benchmark dataset has anonymised
features that make semantically meaningful explanation impossible in principle.

### 4.2 Connection to the proposed research

These gaps shape the thesis directly. Rather than adding a further instance of the
standard pipeline, the proposed work takes the explanation stage itself as its object of
study, with three contributions.

**First, direct measurement of explanation quality.** SHAP and LIME explanations of a
fraud model will be evaluated against measurable properties — fidelity to the underlying
model, stability under small input perturbations, and agreement between methods — rather
than being accepted on the strength of the model's accuracy. This addresses Gap 1 and
operationalises **RQ2**.

**Second, a controlled test of the imbalance–explanation interaction.** Identical model
architectures will be trained under several imbalance regimes (natural, random
undersampling, SMOTE), with explanations compared across them. If the explainability–
imbalance paradox is real, explanations should diverge systematically even where
predictive performance does not. This tests a currently unverified claim and addresses
Gap 2 and **RQ3**.

**Third, evaluation on semantically meaningful features.** At least one dataset with
interpretable attributes will be used alongside the ULB benchmark, allowing the practical
question — can an investigator act on this explanation? — to be asked at all. This
addresses Gap 4.

The resulting contribution is evidential rather than architectural. The field does not
presently need another SHAP-plus-XGBoost fraud detector; it needs evidence about whether
the explanations such systems already produce can withstand scrutiny. That is a question
answerable at master's scale with public data, and its answer is useful whichever way it
falls: confirmation would strengthen current practice, and refutation would identify a
methodological fault affecting a large and growing body of work.

---

**Word count (Sections 1–4, excluding headings, header block and references): 2,459 words.**

---

## References

Adadi, A., & Berrada, M. (2018). Peeking inside the black-box: A survey on explainable artificial intelligence (XAI). *IEEE Access*, 6, 52138–52160.

Baisholan, N., Dietz, J. E., Gnatyuk, S., et al. (2025). FraudX AI: An interpretable machine learning framework for credit card fraud detection on imbalanced datasets. *Computers*, 14(4), 120.

Barredo Arrieta, A., Díaz-Rodríguez, N., Del Ser, J., et al. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion*, 58, 82–115.

Bhatt, U., Xiang, A., Sharma, S., et al. (2020). Explainable machine learning in deployment. *Proceedings of FAccT 2020*, 648–657.

Černevičienė, J., & Kabašinskas, A. (2024). Explainable artificial intelligence (XAI) in finance: A systematic literature review. *Artificial Intelligence Review*, 57.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. *IEEE Symposium Series on Computational Intelligence*, 159–166.

Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. *arXiv:1702.08608*.

*Explainable artificial intelligence models for detecting suspicious bank transactions* (2026). *International Journal of Machine Learning and Cybernetics*. https://doi.org/10.1007/s13042-026-02994-w

Gupta, D., Chugh, P., Sood, K., & Grima, S. (2026). Explainable artificial intelligence in financial fraud detection: A systematic review and FinTech-oriented ADO–TCCM meta-framework for trust, governance, and transparency. *FinTech*, 5(3), 60.

Jesus, S., Belém, C., Balayan, V., Bento, J., Saleiro, P., Bizarro, P., & Gama, J. (2021). How can I choose an explainer? An application-grounded evaluation of post-hoc explanations. *Proceedings of FAccT 2021*, 805–815.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems* 30, 4765–4774.

*Methodological challenges in explainable AI for fraud detection: A systematic literature review* (2025). *Research Square*, rs-7382613/v1. **[Preprint — not peer reviewed.]**

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *Proceedings of KDD 2016*, 1135–1144.

Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206–215.

Slack, D., Hilgard, S., Jia, E., Singh, S., & Lakkaraju, H. (2020). Fooling LIME and SHAP: Adversarial attacks on post hoc explanation methods. *Proceedings of AIES 2020*, 180–186.

Wachter, S., Mittelstadt, B., & Floridi, L. (2017). Why a right to explanation of automated decision-making does not exist in the General Data Protection Regulation. *International Data Privacy Law*, 7(2), 76–99.

Weerts, H. J. P., van Ipenburg, W., & Pechenizkiy, M. (2019). A human-grounded evaluation of SHAP for alert processing. *arXiv:1907.03324*.

Zhou, Y., et al. (2023). A user-centered explainable artificial intelligence approach for financial fraud detection. *Finance Research Letters*, 58, 104309.

*AI-based credit card fraud detection: A machine learning approach with model explainability on real-world data* (2026). *Knowledge and Information Systems*. https://doi.org/10.1007/s10115-026-02773-7
