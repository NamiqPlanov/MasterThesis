"""
Build a standalone description of the thesis project.

A self-contained account of the work for anyone who needs to understand it
without reading the thesis: the problem, the data, the models, the measurement
framework, the methodological decisions and why they were made, the results,
and the limitations. No academic apparatus and no reader-specific framing.

Numbers are hard-coded rather than read from results/ because this document is
a snapshot. Every figure in it was taken from results/ on 15 November 2026.

    .venv\\Scripts\\python.exe reports\\build_project_overview.py
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "Project_Overview-NamigPlanov.docx"

FONT = "Times New Roman"
BODY_PT = 11
HEADER_FILL = "EDEDE6"


def force_font(style, family=FONT):
    """Pin a style to a real font, theme attributes included.

    Word names the built-in heading fonts through the theme
    (w:asciiTheme="majorHAnsi"), and that overrides a plain w:ascii, so it
    has to be removed or headings silently render in Calibri.
    """
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.insert(0, rf)
    for themed in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        rf.attrib.pop(qn("w:" + themed), None)
    for direct in ("ascii", "hAnsi", "eastAsia", "cs"):
        rf.set(qn("w:" + direct), family)


def setup(doc):
    for name, size, bold, italic, before in (
            ("Normal", BODY_PT, False, False, 0),
            ("Heading 1", 13, True, False, 13),
            ("Heading 2", 11, True, False, 9),
            ("List Bullet", BODY_PT, False, False, 0)):
        st = doc.styles[name]
        st.font.name = FONT
        st.font.size = Pt(size)
        st.font.bold = bold
        st.font.italic = italic
        st.font.color.rgb = RGBColor(0, 0, 0)
        force_font(st)
        pf = st.paragraph_format
        pf.line_spacing = 1.15
        pf.space_before = Pt(before)
        pf.space_after = Pt(4)
        if name.startswith("Heading"):
            pf.keep_with_next = True
    sec = doc.sections[0]
    for attr in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(sec, attr, Inches(1))
    foot = sec.footer.paragraphs[0]
    foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = foot.add_run()
    run.font.name, run.font.size = FONT, Pt(9)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.addnext(fld)


def para(doc, parts, style=None):
    p = doc.add_paragraph(style=style)
    if isinstance(parts, str):
        parts = [(parts, False, False)]
    for text, bold, italic in parts:
        r = p.add_run(text)
        r.bold, r.italic = bold or None, italic or None
    return p


def shade(cell, fill):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:color"), "auto")
    el.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(el)


def table(doc, widths, header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold, r.font.size, r.font.name = True, Pt(9.5), FONT
        c.paragraphs[0].paragraph_format.line_spacing = 1.0
        c.paragraphs[0].paragraph_format.space_after = Pt(0)
        shade(c, HEADER_FILL)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            c = cells[i]
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.size, r.font.name = Pt(9.5), FONT
    for r in t.rows:
        for i, w in enumerate(widths):
            r.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


# ============================================================ build =========
doc = Document()
setup(doc)

for text, size, bold in (
        ("Explainable AI for Fraud Detection in Financial Transactions",
         13, True),
        ("Project Overview", 11, False),
        ("Namig Planov · MSc, School of IT and Engineering, "
         "ADA University · Supervisor: Abdul Razaque", 10, False)):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.bold, r.font.size = bold or None, Pt(size)

rule = doc.add_paragraph()
rule.paragraph_format.space_after = Pt(8)
pbdr = OxmlElement("w:pBdr")
bot = OxmlElement("w:bottom")
for k, v in (("w:val", "single"), ("w:sz", "6"), ("w:space", "1"),
             ("w:color", "999999")):
    bot.set(qn(k), v)
pbdr.append(bot)
rule._p.get_or_add_pPr().append(pbdr)

# ---- 1 ---------------------------------------------------------------------
doc.add_paragraph("1. Overview", style="Heading 1")

para(doc, [
    ("Banks detect payment fraud with machine-learning models that are "
     "accurate but opaque. Because a flagged transaction leads to a declined "
     "payment, a frozen account or a report to a financial intelligence "
     "unit, both regulators and institutions require those decisions to be "
     "explained. The industry's response has been to attach post-hoc "
     "explanation tools — principally SHAP and LIME — to the "
     "models and treat their output as trustworthy.", False, False)])

para(doc, [
    ("This project asks whether those explanations withstand direct "
     "measurement. ", False, False),
    ("The object of study is the explanation itself, not the model", True, False),
    (". Three fraud detectors are trained on two public datasets, SHAP and "
     "LIME explanations are generated for the transactions a human analyst "
     "would actually review, and four properties of those explanations are "
     "measured: whether they are faithful to the model, stable under small "
     "changes, in agreement with one another, and actionable for an "
     "investigator. A controlled experiment additionally tests whether the "
     "standard remedy for class imbalance distorts them.", False, False)])

para(doc, [
    ("The central finding is that explanation quality and detection quality "
     "diverge, and sometimes invert. On the dataset with real feature names, "
     "the best-performing detector produces the least stable explanations: "
     "XGBoost attains the highest detection score and simultaneously a "
     "max-sensitivity of 0.960, meaning a 5% perturbation of the input "
     "changes the explanation by almost as much as the explanation's own "
     "magnitude.", False, False)])

# ---- 2 ---------------------------------------------------------------------
doc.add_paragraph("2. Problem and motivation", style="Heading 1")

para(doc, [
    ("Three findings from the literature review motivate the work.",
     False, False)])

para(doc, [
    ("Explanation quality is asserted rather than measured. ", True, False),
    ("A systematic review of 49 studies found that over 80% use a model's "
     "predictive performance as a proxy for the quality of its explanations. "
     "Those are logically independent properties: accuracy cannot serve as "
     "evidence of fidelity.", False, False)], style="List Bullet")

para(doc, [
    ("Where explanations have been tested on people, results were negative. ",
     True, False),
    ("Weerts et al. (2019) found no significant improvement in analysts' "
     "alert-processing performance when SHAP was available, and Jesus et al. "
     "(2021), working with real fraud analysts on a deployed model, concluded "
     "that popular explainers had “a worse impact than desired”. "
     "This is the strongest human evidence in the field and it is seldom "
     "cited by the applied work that assumes the opposite.", False, False)],
     style="List Bullet")

para(doc, [
    ("The field's dominant benchmark cannot demonstrate the claim. ",
     True, False),
    ("The standard credit-card dataset has features named V1–V28, which "
     "are anonymised principal components. An explanation over that data "
     "states that “V14” drove the decision — a statement no "
     "investigator can act on and no customer can be told.", False, False)],
     style="List Bullet")

para(doc, [
    ("A regulatory driver compounds this. The EU AI Act classifies "
     "financial-services systems as high-risk and Article 22 of the GDPR "
     "constrains solely automated decision-making, so institutions are "
     "generating these explanations at scale to satisfy obligations whose "
     "evidential basis has not been examined.", False, False)])

# ---- 3 ---------------------------------------------------------------------
doc.add_paragraph("3. Research objectives", style="Heading 1")

para(doc, "The work pursues four questions:")
for q in (
    "Are post-hoc explanations of fraud models faithful to the models they "
    "describe, and stable under small input perturbations?",
    "Do independent explanation methods, and independently trained models, "
    "agree about why a given transaction is suspicious?",
    "Does class-imbalance handling — intrinsic to fraud data — "
    "systematically distort explanations while leaving detection unchanged?",
    "Can an investigator act on the explanations produced, and does a "
    "semantically meaningful feature space suffice to make them actionable?",
):
    para(doc, q, style="List Bullet")

para(doc, [
    ("The contribution is evidential rather than architectural. The project "
     "does not propose a new detector; it provides evidence about whether "
     "the explanations existing systems produce can withstand scrutiny.",
     False, False)])

# ---- 4 ---------------------------------------------------------------------
doc.add_paragraph("4. Data", style="Heading 1")

table(doc, [1.25, 2.5, 2.52],
      ["", "ULB Credit Card", "IEEE-CIS (Vesta Corporation)"],
      [["Size", "284,807 transactions", "590,540 labelled; a 300,000-row "
        "chronological prefix is used, for memory reasons"],
       ["Fraud rate", "492 frauds, 0.172%", "3.37% within the prefix"],
       ["Span", "48 hours, September 2013",
        "182 days in full; 85 days within the prefix"],
       ["Features", "30, of which 28 are anonymised PCA components "
        "(V1–V28); only Time and Amount are untransformed",
        "390 retained after dropping two columns over 90% empty; 23 are "
        "semantically meaningful"],
       ["Rationale", "The field's dominant benchmark, which makes results "
        "comparable with prior work, and the negative case for actionability",
        "Supplies real feature names — amount, product code, card and "
        "address fields, email domain — so the actionability question "
        "can be asked at all"]])

para(doc, [
    ("The two datasets were chosen for their opposing properties. One is the "
     "benchmark most of the literature uses but which is structurally "
     "incapable of demonstrating meaningful explanation; the other can "
     "demonstrate it. IEEE-CIS's competition test partition is excluded "
     "throughout because its labels are withheld.", False, False)])

para(doc, [
    ("Both datasets are public and de-identified at source, and neither is "
     "redistributed: the ULB data is licensed under the Open Database "
     "License and the IEEE-CIS data under the terms of its Kaggle "
     "competition.", False, False)])

# ---- 5 ---------------------------------------------------------------------
doc.add_paragraph("5. Models", style="Heading 1")

para(doc, [
    ("The three classifiers are not competing for deployment; they are ",
     False, False),
    ("experimental conditions", True, False),
    (" spanning the interpretability–performance range the literature "
     "contrasts, so that the contrast can be tested rather than assumed.",
     False, False)])

table(doc, [1.45, 2.25, 2.57],
      ["Model", "Configuration", "Role"],
      [["Logistic regression",
        "balanced class weights, max_iter 1,000–3,000",
        "The interpretable anchor. Additive and directly inspectable, so it "
        "approximates the interpretable-by-design alternative."],
       ["Random forest",
        "300 trees on ULB, 150 on IEEE-CIS; min_samples_leaf 2–5; "
        "balanced subsample weighting",
        "The field's most common choice, representing current practice."],
       ["XGBoost",
        "400 trees, depth 6, learning rate 0.1, subsample 0.9, "
        "colsample_bytree 0.9, scale_pos_weight set to the "
        "negative-to-positive ratio",
        "Gradient boosting dominates published fraud benchmarks, so it "
        "represents the model most likely to be deployed."]])

para(doc, [
    ("The decisive selection criterion was that ", False, False),
    ("every model must have an exact SHAP algorithm available", True, False),
    (" — LinearExplainer for the linear model and TreeExplainer for the "
     "ensembles. A model requiring the sampling-based KernelExplainer would "
     "confound the explainer's own approximation error with the property "
     "being measured, making an unstable explanation indistinguishable from "
     "an unstable estimate of a stable one. Deep tabular architectures were "
     "therefore excluded on measurement grounds rather than performance "
     "grounds.", False, False)])

para(doc, [
    ("Hyperparameters are fixed a priori at conventional values and "
     "deliberately not tuned, because a tuning search would introduce a "
     "selection process whose own interaction with explanation quality is "
     "unmeasured. The models are consequently reasonable rather than "
     "optimal, which is stated as a limitation.", False, False)])

# ---- 6 ---------------------------------------------------------------------
doc.add_paragraph("6. Explanation methods", style="Heading 1")

para(doc, [
    ("SHAP", True, False),
    (" (Lundberg & Lee, 2017) assigns each feature a contribution to an "
     "individual prediction, derived from Shapley values in cooperative game "
     "theory. The contributions sum exactly to the difference between that "
     "prediction and the average prediction. Exact polynomial-time "
     "algorithms exist for linear models and tree ensembles; other model "
     "classes require sampling.", False, False)])

para(doc, [
    ("LIME", True, False),
    (" (Ribeiro et al., 2016) is empirical rather than axiomatic. It "
     "perturbs the input many times, weights each perturbation by its "
     "proximity to the original, and fits a simple linear surrogate to that "
     "local neighbourhood; the surrogate's coefficients are the explanation. "
     "Because the surrogate is an empirical fit, it can fail without raising "
     "an error — a failure mode this project encountered and measured.",
     False, False)])

para(doc, [
    ("Explanations are generated for two populations, which answer different "
     "questions. The ", False, False),
    ("alert queue", False, True),
    (" is the 400 highest-scored test transactions, representing what an "
     "analyst actually sees. The ", False, False),
    ("random sample", False, True),
    (" is 600 further transactions drawn from the remainder, representing "
     "the population the model scores. Together they give 1,000 instances "
     "per configuration.", False, False)])

# ---- 7 ---------------------------------------------------------------------
doc.add_paragraph("7. Measurement framework", style="Heading 1")

table(doc, [1.35, 2.4, 2.52],
      ["Property", "Measurement", "Interpretation"],
      [["Fidelity",
        "Delete the 20 highest-ranked features one at a time, replacing each "
        "with its training-set mean, and record the fall in predicted fraud "
        "probability",
        "Whether the explanation names features the model is actually using. "
        "A near-zero score means it named features the model ignores."],
       ["Stability (input)",
        "Max-sensitivity: perturb the input within 5% of each feature's "
        "magnitude, eight times, and take the largest relative change in the "
        "attribution vector",
        "Whether the explanation would survive a slightly different "
        "transaction. Lower is better."],
       ["Stability (training)",
        "Train ten models differing only in random seed and compare their "
        "explanations of the same fixed alert queue",
        "Whether the explanation is a property of the data or an artefact of "
        "one training run."],
       ["Agreement",
        "Spearman rank correlation across all features, and Jaccard overlap "
        "of the top five",
        "Whether two explainers, or two models, tell the same story about one "
        "transaction. The two measures can diverge sharply."],
       ["Actionability",
        "Share of alerts whose highest-attributed feature is semantically "
        "meaningful rather than an anonymised engineered one",
        "Whether an investigator could pursue the explanation."]])

para(doc, [
    ("The measures were validated before use on synthetic models whose true "
     "feature importances are known by construction. Seven checks pass, "
     "including the control that a ", False, False),
    ("random", False, True),
    (" attribution fails recovery. A measure that cannot recover a known "
     "answer cannot be trusted on an unknown one.", False, False)])

para(doc, [
    ("Inference uses 95% percentile bootstrap confidence intervals over "
     "2,000 resamples and Wilcoxon signed-rank tests for paired "
     "comparisons. Because 400 paired instances make every p-value "
     "approximately 10⁻⁶⁷, Cohen's d is reported alongside "
     "every test and carries the substantive argument.", False, False)])

# ---- 8 ---------------------------------------------------------------------
doc.add_paragraph("8. Methodological decisions", style="Heading 1")

doc.add_paragraph("8.1 Chronological splits rather than random",
                  style="Heading 2")
para(doc, [
    ("Transactions are sorted by timestamp, with the first 70% used for "
     "training and the last 30% for testing. Most published work uses a "
     "random split. Fraud is not stationary — measured across "
     "chronological deciles the fraud rate varies several-fold in both "
     "datasets — so a random split permits a model to learn from "
     "transactions occurring after those it is evaluated on. Reported "
     "performance here is consequently lower than the literature's, and that "
     "gap estimates how much the standard protocol flatters itself.",
     False, False)])

doc.add_paragraph("8.2 A fixed alert budget rather than a probability "
                  "threshold", style="Heading 2")
para(doc, [
    ("An ", False, False),
    ("alert budget", True, False),
    (" b is the share of transaction volume an investigation team has "
     "capacity to review. The alerts at budget b are the top b fraction of "
     "test transactions ranked by model score, so the budget fixes the "
     "number of cases reviewed and the model determines only which cases "
     "fill them. Budgets of 0.1%, 0.5% and 1% are reported; at 0.5% that is "
     "427 alerts on ULB and 450 on IEEE-CIS.", False, False)])
para(doc, [
    ("The construct is operational. A fraud team employs a fixed number of "
     "analysts who can each clear a fixed number of cases per shift, so "
     "review capacity is approximately constant and independent of how much "
     "fraud is present. A probability threshold does not have that property: "
     "the same threshold yields different alert volumes on different days, "
     "and a poorly calibrated model can exhaust a team's capacity without "
     "ranking any better.", False, False)])

doc.add_paragraph("8.3 AUC-PR rather than accuracy", style="Heading 2")
para(doc, [
    ("At a 0.172% base rate, a classifier answering “legitimate” "
     "for every transaction is 99.83% accurate and catches nothing. Accuracy "
     "is therefore not reported at all. AUC-PR is the primary "
     "threshold-free measure, in preference to ROC-AUC, which is dominated "
     "by the large negative class and compresses the differences that "
     "matter. AUC-PR alone is also insufficient: on ULB the random forest "
     "attains the best AUC-PR while logistic regression wins both precision "
     "and recall at the 0.5% budget.", False, False)])

doc.add_paragraph("8.4 Imbalance handled by weighting in the baselines",
                  style="Heading 2")
para(doc, [
    ("All three baseline models address class imbalance through instance "
     "weighting — class_weight in scikit-learn, scale_pos_weight in "
     "XGBoost — rather than resampling, because resampling is the "
     "manipulated variable of the imbalance experiment. Using it in the "
     "baseline would confound treatment with control.", False, False)])

doc.add_paragraph("8.5 Adaptations to the standard metrics", style="Heading 2")
para(doc, [
    ("Deletion fidelity in place of insertion-minus-deletion. ", True, False),
    ("An insertion curve filling only the top 20 of 390 features never "
     "approaches the true prediction, which drives that measure negative for "
     "reasons unrelated to explanation quality.", False, False)],
     style="List Bullet")
para(doc, [
    ("Absolute rather than normalised fidelity. ", True, False),
    ("Dividing the probability drop by the starting prediction explodes on "
     "low-scored transactions, where a ULB model outputs approximately "
     "10⁻⁶ and any movement becomes a ratio in the thousands.",
     False, False)], style="List Bullet")
para(doc, [
    ("Batched perturbations in the stability measure. ", True, False),
    ("Explaining each perturbation separately rebuilt the explainer every "
     "time; batching all perturbations of an instance into one call is what "
     "made the measurement feasible on a 390-feature forest.", False, False)],
     style="List Bullet")
para(doc, [
    ("Per-instance selection of LIME's kernel width. ", True, False),
    ("LIME's default locality kernel is degenerate at 390 features, for the "
     "reasons given in section 10.2. The width is therefore chosen for each "
     "instance by maximising the surrogate's own goodness of fit, and "
     "instances whose surrogate still fails to fit are excluded and counted.",
     False, False)], style="List Bullet")

# ---- 9 ---------------------------------------------------------------------
doc.add_paragraph("9. Results", style="Heading 1")

doc.add_paragraph("9.1 Detection performance", style="Heading 2")
table(doc, [1.05, 1.85, 0.85, 0.85, 0.82, 0.85],
      ["Dataset", "Model", "AUC-PR", "ROC-AUC", "Prec@0.5%", "Rec@0.5%"],
      [["ULB", "Logistic regression", "0.7677", "0.9796", "0.218", "0.861"],
       ["ULB", "Random forest", "0.8196", "0.9529", "0.211", "0.833"],
       ["ULB", "XGBoost", "0.8050", "0.9773", "0.211", "0.833"],
       ["IEEE-CIS", "Logistic regression", "0.4139", "0.8384", "0.913", "0.109"],
       ["IEEE-CIS", "Random forest", "0.5154", "0.8845", "0.980", "0.117"],
       ["IEEE-CIS", "XGBoost", "0.5756", "0.8882", "0.984", "0.118"]])

para(doc, [
    ("The two datasets appear to fail in opposite directions: ULB catches "
     "86% of fraud while being wrong about four alerts in five, whereas "
     "IEEE-CIS is correct about 98% of its alerts and misses seven frauds in "
     "eight. ", False, False),
    ("This difference is largely arithmetic rather than a difference in "
     "model quality.", True, False),
    (" With k alerts and F frauds at most min(k, F) frauds can be caught, so "
     "precision is bounded by F/k and recall by k/F. ULB has 427 alerts for "
     "108 frauds — roughly four slots per fraud, so recall can be high "
     "and precision cannot exceed 0.253. IEEE-CIS has 450 alerts for 3,759 "
     "frauds — one slot per eight, so precision can approach 1.0 and "
     "recall cannot exceed 0.120.", False, False)])

para(doc, [
    ("Dividing each observed rate by its ceiling yields the same quantity, "
     "caught / min(k, F). ULB attains 83–86% of what is achievable and "
     "IEEE-CIS 91–98%. Normalised this way, the IEEE-CIS models are the "
     "stronger ones — the reverse of what the raw rates suggest.",
     False, False)])

doc.add_paragraph("9.2 Explanation quality", style="Heading 2")
table(doc, [0.98, 1.72, 1.0, 1.0, 0.78, 0.79],
      ["Dataset", "Model", "Fidelity (alerts)", "Fidelity (random)",
       "Max-sens.", "SHAP vs LIME"],
      [["ULB", "Logistic regression", "0.645", "0.032", "0.039", "0.651"],
       ["ULB", "Random forest", "0.170", "−0.000", "0.061", "0.511"],
       ["ULB", "XGBoost", "0.218", "−0.000", "0.123", "0.309"],
       ["IEEE-CIS", "Logistic regression", "0.127", "0.011", "0.038", "0.413"],
       ["IEEE-CIS", "Random forest", "0.155", "−0.052", "0.204", "0.399"],
       ["IEEE-CIS", "XGBoost", "0.113", "−0.100", "0.960", "0.231"]])

para(doc, [
    ("The interpretable model has the most faithful explanations. ", True, False),
    ("Deleting the 20 features SHAP named reduces logistic regression's ULB "
     "fraud score by two thirds (0.645), against 0.170 and 0.218 for the "
     "ensembles. SHAP is exact for all three model classes, so this is not "
     "explainer error; an ensemble's prediction is distributed across feature "
     "interactions that no single-feature deletion dislodges.", False, False)],
     style="List Bullet")
para(doc, [
    ("Stability degrades with dimensionality, but only for the ensembles. ",
     True, False),
    ("Logistic regression is essentially dimension-independent at 0.039 and "
     "0.038 across 30 and 390 features, while XGBoost moves from 0.123 to "
     "0.960 and the random forest from 0.061 to 0.204.", False, False)],
     style="List Bullet")
para(doc, [
    ("Away from the alert queue, attributions are anti-faithful. ", True, False),
    ("On IEEE-CIS both tree ensembles return negative deletion fidelity on "
     "the random sample — −0.052 and −0.100, with confidence "
     "intervals excluding zero — meaning that deleting the features SHAP "
     "ranked highest moves the fraud score upward. Median scores there are "
     "0.045 to 0.339, so this is not an absence of probability mass to "
     "remove. The consequence falls on global explanation, because a SHAP "
     "summary plot aggregated over a representative sample is dominated by "
     "exactly this population.", False, False)], style="List Bullet")
para(doc, [
    ("Rank correlation can conceal complete disagreement. ", True, False),
    ("On IEEE-CIS, logistic regression and the random forest reach "
     "ρ = 0.553, which a study reporting correlation alone would "
     "describe as substantial agreement, while their top-five Jaccard "
     "overlap is 0.013. Across 400 alerts the two models almost never "
     "nominate a shared feature among the five an analyst would read; they "
     "order the long tail of irrelevant features similarly and disagree "
     "about everything that matters.", False, False)], style="List Bullet")

doc.add_paragraph("9.3 The imbalance–explanation interaction",
                  style="Heading 2")
para(doc, [
    ("One architecture is trained under three regimes — natural "
     "imbalance, random undersampling and SMOTE — on one split, with "
     "one test set, each model explaining the same fixed alert queue. On ULB "
     "detection quality barely moves (AUC-PR spans 0.0244) while explanation "
     "agreement with the natural-imbalance model falls to ρ = 0.446 "
     "under undersampling, at a Cohen's d of 4.00. The most-named feature "
     "changes outright: the natural model attributes 88% of alerts primarily "
     "to V14, while the undersampled model names V4 most often. Two models a "
     "practitioner would treat as interchangeable on detection quality "
     "therefore tell materially different stories about the same "
     "transactions.", False, False)])

para(doc, [
    ("A control experiment rules out the obvious alternative explanation, "
     "that SHAP attributions are simply noisy. Ten models differing only in "
     "random seed, each explaining the same alert queue, agree at a mean "
     "ρ of 0.873 with AUC-PR varying by only 0.0134. Retraining moves "
     "the ranking roughly a tenth as far as resampling does, so the "
     "divergence is attributable to the imbalance strategy rather than to "
     "stochastic training.", False, False)])

para(doc, [
    ("On IEEE-CIS the covariate is held less well — AUC-PR spans "
     "0.0624, with SMOTE improving the natural regime from 0.5100 to 0.5724 "
     "— so part of the divergence there may reflect a genuinely better "
     "model. The ULB result, where performance is effectively flat, carries "
     "the argument.", False, False)])

doc.add_paragraph("9.4 Semantic actionability", style="Heading 2")
para(doc, [
    ("Of the 390 IEEE-CIS features retained, 23 are semantically "
     "meaningful, yet all three models rank an opaque feature first for "
     "between 97.5% and 99.8% of alerts: the top-ranked feature is semantic "
     "for 0.2% of alerts under logistic regression, 2.0% under the random "
     "forest and 2.5% under XGBoost. XGBoost names the counting feature C1 "
     "first for 66% of its alerts. ULB scores zero by construction. A "
     "semantically meaningful feature space is therefore necessary but not "
     "sufficient for an actionable explanation: the names must exist, and the "
     "model must use them.", False, False)])

# ---- 10 --------------------------------------------------------------------
doc.add_paragraph("10. Engineering considerations", style="Heading 1")

doc.add_paragraph("10.1 Recording experimental conditions", style="Heading 2")
para(doc, [
    ("The stability metric was initially sampling 80 instances where the "
     "research proposal commits to at least 1,000, and because no script "
     "recorded its sample size the discrepancy was invisible in the results "
     "files. At 80 instances the two ULB tree models had overlapping "
     "confidence intervals and appeared equally unstable; at 1,000 the "
     "intervals are disjoint and XGBoost is twice as unstable as the random "
     "forest. On IEEE-CIS the XGBoost estimate moved from 0.603 to 0.960. "
     "Every script now writes its seed, instance counts, fidelity curve "
     "length and top-k value into its results file, so a condition mismatch "
     "cannot remain hidden.", False, False)])

doc.add_paragraph("10.2 A silent failure in LIME at high dimensionality",
                  style="Heading 2")
para(doc, [
    ("LIME weights each perturbation by exp(−D²/w²), with w "
     "defaulting to √d · 0.75. At 390 features, on an alert scored "
     "0.99996, the model's response surface is locally flat, so a narrow "
     "kernel draws only perturbations whose output is identical, the "
     "weighted regression has no variance to fit, and the returned "
     "coefficients are numerical noise of order 10⁻²⁹. The "
     "surrogate's goodness of fit was 0.006 and no error was raised. "
     "Increasing the sample count does not repair it — the fit remains "
     "near 0.001 at 10,400 samples — because the problem is the "
     "weighting rather than the sampling.", False, False)])
para(doc, [
    ("Widening the kernel does repair it, but the required width is a "
     "property of the individual transaction rather than of the dataset: "
     "within a single alert queue the best multiplier varies. The width is "
     "therefore selected per instance by maximising the surrogate's local "
     "fit, a criterion that depends only on the LIME fit and never on the "
     "SHAP vector it is later compared against, with instances below a fit "
     "of 0.5 excluded and counted. This raises the median fit to between "
     "0.826 and 0.998. The same procedure applied to ULB selects the default "
     "width for 99 to 102 of 120 instances, which is the intended behaviour: "
     "one implementation adapts to both dimensionalities rather than being "
     "tuned for one.", False, False)])
para(doc, [
    ("The general lesson is that matching a hyperparameter across "
     "experimental conditions is not the same as matching the condition "
     "itself.", False, False)])

doc.add_paragraph("10.3 Numerical precision", style="Heading 2")
para(doc, [
    ("Probability calibration required clipping predicted probabilities away "
     "from 0 and 1 before taking a logit. Clipping to 1 − 10⁻¹² "
     "failed for one model, whose predict_proba returns float32: that bound "
     "is not representable in float32 and rounds to exactly 1.0, so the clip "
     "silently did nothing, 150 scores remained at exactly 1.0, and the "
     "logit became infinite. Casting to float64 before clipping resolves it.",
     False, False)])

doc.add_paragraph("10.4 Computational cost", style="Heading 2")
para(doc, [
    ("The heaviest experiment required roughly five hours of processor time, "
     "measuring stability over 1,000 instances against a 390-feature random "
     "forest. Three measures addressed cost: batching perturbations into a "
     "single explainer call; subsampling IEEE-CIS to a chronological "
     "300,000-row prefix because the full file exhausted available memory, "
     "taking a prefix rather than a random draw so that time ordering "
     "survives; and silencing a log that reached 88 MB because joblib's "
     "worker processes do not inherit Python's warning filters, whereas the "
     "PYTHONWARNINGS environment variable is inherited.", False, False)])

# ---- 11 --------------------------------------------------------------------
doc.add_paragraph("11. Tools", style="Heading 1")
table(doc, [1.55, 4.72],
      ["Area", "Technology"],
      [["Language", "Python 3.11"],
       ["Modelling", "scikit-learn (logistic regression, random forest), "
        "XGBoost, imbalanced-learn (SMOTE, RandomUnderSampler)"],
       ["Explanation", "shap (TreeExplainer, LinearExplainer), lime "
        "(LimeTabularExplainer)"],
       ["Statistics", "SciPy (Wilcoxon signed-rank, Spearman, "
        "Kolmogorov–Smirnov), percentile bootstrap, Cohen's d"],
       ["Data", "pandas, NumPy; PostgreSQL 18.4 for exploratory querying, "
        "loaded via COPY FROM STDIN"],
       ["Output", "Matplotlib (300 dpi figures, colour-vision-safe "
        "palette), python-docx"],
       ["Engineering", "Git, joblib for model persistence and parallelism, "
        "fixed random seeds throughout"]])

para(doc, [
    ("One practice is worth noting. Every experiment writes its results to a "
     "JSON file, and all summary tables and figures are generated from those "
     "files rather than transcribed, so a reported value cannot disagree with "
     "the experiment that produced it. This was introduced after a figure "
     "caption was found to contradict the chart above it.", False, False)])

# ---- 12 --------------------------------------------------------------------
doc.add_paragraph("12. Limitations", style="Heading 1")

para(doc, [
    ("The principal limitation is deliberate. The strongest missing evidence "
     "in this field concerns whether explanations help ", False, False),
    ("professional fraud analysts", True, False),
    (", and recruiting them was not feasible within a master's project. This "
     "work therefore measures proxy properties, which are ", False, False),
    ("necessary but not sufficient", False, True),
    (": an explanation that is unfaithful or unstable cannot be useful, but "
     "one that is faithful and stable is not thereby proven useful. The "
     "findings constrain the field's claims rather than settling them.",
     False, False)])

for lim in (
    "IEEE-CIS experiments use a 300,000-row chronological subsample, so "
    "conclusions about that dataset strictly concern its first 85 days.",
    "Hyperparameters are untuned, so the models are reasonable rather than "
    "optimal.",
    "The imbalance experiment manipulates a single architecture, so "
    "generalisation across model families is not established.",
    "Budget-level precision and recall are reported without confidence "
    "intervals.",
    "The qualitative half of the actionability study requires a second "
    "independent rater and is outstanding.",
):
    para(doc, lim, style="List Bullet")

# ---- 13 --------------------------------------------------------------------
doc.add_paragraph("13. Key terms", style="Heading 1")
table(doc, [1.7, 4.57],
      ["Term", "Definition"],
      [["AUC-PR", "Area under the precision-recall curve; a threshold-free "
        "measure that is not dominated by the negative class at low base "
        "rates."],
       ["Alert budget", "The share of transaction volume an investigation "
        "team can review, which fixes the number of alerts so that only the "
        "ranking varies."],
       ["Precision / recall at budget", "Of the reviewed cases, the share "
        "that were fraudulent; and of all fraud present, the share that "
        "reached the review queue."],
       ["Fidelity", "Whether an explanation names the features the model is "
        "actually using."],
       ["Max-sensitivity", "The largest change in an explanation produced by "
        "a small change in the input; lower is better."],
       ["SMOTE", "Synthetic Minority Over-sampling Technique; generates new "
        "minority examples by interpolating between real ones, which can "
        "place points away from the data manifold."],
       ["Cohen's d", "A standardised effect size: the magnitude of a "
        "difference expressed in units of its own variability, independent of "
        "sample size."],
       ["Bootstrap confidence interval", "An interval obtained by resampling "
        "the data with replacement many times and taking percentiles of the "
        "resulting estimates."],
       ["Wilcoxon signed-rank test", "A non-parametric paired test, "
        "appropriate where per-instance scores are bounded and not normally "
        "distributed."],
       ["Concept drift", "Change over time in the relationship between "
        "features and the target, which motivates chronological splits."],
       ["Calibration", "Whether a predicted probability of 0.8 corresponds "
        "to an 80% observed rate; measured by Brier score and expected "
        "calibration error."]])

doc.save(OUT)
print(f"wrote {OUT.relative_to(ROOT)}")
print(f"  paragraphs {len(doc.paragraphs)}  tables {len(doc.tables)}")
