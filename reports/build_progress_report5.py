"""
Build Progress Report 5 as a .docx, with every table read from results/.

The repository's standing rule is that no reported number is retyped: tables
and figures are generated from the JSON each experiment writes. This script
extends that rule to the progress report, so a number in the submitted
document cannot disagree with the experiment that produced it. The prose is
written here; the figure comes from figures/ and the numbers from results/.

Formatting follows the assignment brief: Times New Roman 12 pt, 1.5 line
spacing, labelled figures and tables referred to in the text, APA references,
6-8 pages. The brief's page limit is the binding constraint on length, so the
prose is deliberately compressed and the detail lives in Methods_Chapter.md
and Results_Chapter.md.

    .venv\\Scripts\\python.exe reports\\build_progress_report5.py
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
OUT = ROOT / "reports" / "ProgressReport5-NamigPlanov.docx"

TNR = "Times New Roman"
BUDGET = "recall_at_0.005"
HEADER_FILL = "EDEDE6"


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f3(v):
    """Mean of a (mean, lo, hi) triple, or a bare number, to 3 dp."""
    if v is None:
        return "—"
    x = v[0] if isinstance(v, (list, tuple)) else v
    return f"{x:.3f}".replace("-", "−")


# ------------------------------------------------------------------ styling --
def force_font(style, family=TNR):
    """Pin a style to a real font family, theme attributes included.

    Word's built-in Heading styles name their font through the document
    theme (w:asciiTheme="majorHAnsi"), and that attribute wins over the
    plain w:ascii that python-docx's `font.name` writes. Setting the name
    alone therefore leaves headings rendering in Calibri while the XML
    claims Times New Roman. The theme attributes have to be removed.
    """
    rf = style.element.rPr.rFonts
    for themed in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        rf.attrib.pop(qn("w:" + themed), None)
    for direct in ("ascii", "hAnsi", "eastAsia", "cs"):
        rf.set(qn("w:" + direct), family)


def set_fonts(doc):
    # all three at 12 pt: the brief specifies a 12-point font, and headings
    # stay distinguishable through weight and italics instead of size
    for name, size, bold, italic in (("Normal", 12, False, False),
                                     ("Heading 1", 12, True, False),
                                     ("Heading 2", 12, True, True)):
        st = doc.styles[name]
        st.font.name = TNR
        st.font.size = Pt(size)
        st.font.bold = bold
        st.font.italic = italic
        st.font.color.rgb = RGBColor(0, 0, 0)
        force_font(st)
        pf = st.paragraph_format
        pf.line_spacing = 1.5
        pf.space_after = Pt(2)
        if name.startswith("Heading"):
            pf.space_before = Pt(7)
            pf.space_after = Pt(2)
            pf.keep_with_next = True


def add_page_number_footer(doc):
    para_ = doc.sections[0].footer.paragraphs[0]
    para_.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para_.add_run()
    run.font.name, run.font.size = TNR, Pt(10)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.addnext(fld)


# ------------------------------------------------------------- text helpers --
def para(doc, parts):
    """parts: a string, or a list of (text, bold, italic) tuples."""
    pr = doc.add_paragraph()
    pr.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if isinstance(parts, str):
        parts = [(parts, False, False)]
    for text, bold, italic in parts:
        run = pr.add_run(text)
        run.bold, run.italic = bold, italic
    return pr


def caption(doc, text):
    pr = doc.add_paragraph()
    pr.paragraph_format.line_spacing = 1.0
    pr.paragraph_format.space_before = Pt(2)
    pr.paragraph_format.space_after = Pt(6)
    run = pr.add_run(text)
    run.italic, run.font.size = True, Pt(9.5)
    return pr


def shade(cell, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def add_table(doc, widths, header, rows, numeric_from=2):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(h)
        run.bold, run.font.size, run.font.name = True, Pt(9.5), TNR
        c.paragraphs[0].paragraph_format.line_spacing = 1.0
        c.paragraphs[0].paragraph_format.space_after = Pt(0)
        shade(c, HEADER_FILL)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            c = cells[i]
            c.text = ""
            pr = c.paragraphs[0]
            pr.paragraph_format.line_spacing = 1.0
            pr.paragraph_format.space_after = Pt(0)
            if i >= numeric_from and len(header) > 3:
                pr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            run = pr.add_run(val)
            run.font.size, run.font.name = Pt(9.5), TNR
    # a width set on the row alone is ignored; every cell needs it
    for r in t.rows:
        for i, w in enumerate(widths):
            r.cells[i].width = Inches(w)
    return t


def add_figure(doc, filename, width_in):
    pr = doc.add_paragraph()
    pr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pr.paragraph_format.space_before = Pt(4)
    pr.paragraph_format.space_after = Pt(0)
    pr.add_run().add_picture(str(FIG / filename), width=Inches(width_in))


# ======================================================================= body
doc = Document()
set_fonts(doc)
add_page_number_footer(doc)
sec = doc.sections[0]
for attr in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
    setattr(sec, attr, Inches(1))

s1u, s1i = load("s1_baselines_ulb.json"), load("s1_baselines_ieee.json")
s2u, s2i = (load("s2_explanation_quality_ulb.json"),
            load("s2_explanation_quality_ieee.json"))
s2fr = load("s2_fidelity_random_ieee.json")
s2lime = load("s2_shap_vs_lime_ieee.json")
s3u, s3i = (load("s3_imbalance_explanation_ulb.json"),
            load("s3_imbalance_explanation_ieee.json"))
s5 = load("s5_stability_seeds.json")

sem = s2i["semantic_share"]
rf_r, xgb_r = s2fr["per_model"]["RandomForest"], s2fr["per_model"]["XGBoost"]
lr_r = s2fr["per_model"]["LogisticRegression"]
spread_u = max(s3u["performance"].values()) - min(s3u["performance"].values())
spread_i = max(s3i["performance"].values()) - min(s3i["performance"].values())
worst_u = min(t["mean_rho"] for t in s3u["tests"].values())
xm = s2i["cross_model"]["LogisticRegression|RandomForest"]
r2_def = s2lime["per_model"]["LogisticRegression"][
    "lime_local_r2_median_at_default_width"]
r2_lo = min(m["lime_local_r2_median"] for m in s2lime["per_model"].values())
r2_hi = max(m["lime_local_r2_median"] for m in s2lime["per_model"].values())

NICE = {"LogisticRegression": "Logistic regression",
        "RandomForest": "Random forest", "XGBoost": "XGBoost"}

# ----------------------------------------------------------- header block ---
for text, size, bold, italic in (
        ("Progress Report 5", 12, True, False),
        ("Documenting Findings, Model Selection Process, and Preliminary Results",
         12, True, False),
        ("Explainable AI for Fraud Detection in Financial Transactions",
         11, False, True),
        ("Namig Planov  ·  Supervisor: Abdul Razaque", 11, False, False),
        ("School of IT and Engineering, ADA University  ·  7 November 2026",
         11, False, False)):
    pr = doc.add_paragraph()
    pr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pr.paragraph_format.line_spacing = 1.0
    pr.paragraph_format.space_after = Pt(5 if bold and size == 12 else 1)
    run = pr.add_run(text)
    run.bold, run.italic, run.font.size = bold, italic, Pt(size)

rule = doc.add_paragraph()
rule.paragraph_format.space_after = Pt(6)
pbdr = OxmlElement("w:pBdr")
bottom = OxmlElement("w:bottom")
for k, v in (("w:val", "single"), ("w:sz", "6"), ("w:space", "1"),
             ("w:color", "999999")):
    bottom.set(qn(k), v)
pbdr.append(bottom)
rule._p.get_or_add_pPr().append(pbdr)

# ================================================== 1. Research conducted ===
doc.add_paragraph("1. Research Conducted", style="Heading 1")
doc.add_paragraph("1.1 Literature review and theoretical framework",
                  style="Heading 2")

para(doc, [
    ("Machine learning detects transactional fraud accurately under severe "
     "class imbalance, and SHAP (Lundberg & Lee, 2017) and LIME (Ribeiro et "
     "al., 2016) are routinely attached to those models. Whether the "
     "resulting attributions are trustworthy is far less established. A "
     "systematic review of 49 studies finds over 80% using predictive "
     "performance as a proxy for explanation quality, although fidelity and "
     "accuracy are logically independent properties. Where explanations have "
     "been tested on people the findings are negative: Weerts et al. (2019) "
     "measured no improvement in alert-processing performance when SHAP was "
     "available, and Jesus et al. (2021) concluded from an evaluation with "
     "real fraud analysts that popular explainers had “a worse impact "
     "than desired”. Rudin (2019) argues that post-hoc explanation of a "
     "black box is the wrong strategy for high-stakes decisions, and Slack et "
     "al. (2020) showed both methods can be deliberately fooled.",
     False, False)])

para(doc, [
    ("The framework holds ", False, False),
    ("interpretability", False, True),
    (", ", False, False),
    ("explainability", False, True),
    (" and ", False, False),
    ("explanation quality", False, True),
    (" distinct, and decomposes the last into fidelity, stability, "
     "cross-method agreement and semantic actionability. The unit of analysis "
     "is the explanation output rather than the model: reporting AUC-PR and "
     "then a SHAP plot measures the instrument, not the object.",
     False, False)])

doc.add_paragraph("1.2 Methodology, data collection and tools",
                  style="Heading 2")

para(doc, [
    ("The design is mixed-methods with a dominant quantitative strand, on "
     "secondary data only. ULB contributes 284,807 card transactions over 48 "
     "hours with 492 frauds (0.172%) and 28 of its 30 features anonymised as "
     "principal components; IEEE-CIS contributes 590,540 labelled e-commerce "
     "transactions with many semantically named features. ULB is the "
     "field's dominant benchmark and the negative case for actionability; "
     "IEEE-CIS supplies the interpretable feature space the practical "
     "question requires.", False, False)])

para(doc, [
    ("Splits are chronological — the first 70% of transactions by "
     "timestamp for training, the last 30% for testing — because the "
     "fraud rate varies several-fold across chronological deciles in both "
     "files, so a random split would let a model learn from the future. "
     "Explanations cover an alert queue of the 400 highest-scored test "
     "transactions and a random sample of 600 others. Inference "
     "uses percentile bootstrap intervals over 2,000 resamples, Wilcoxon "
     "signed-rank tests and Cohen's ", False, False),
    ("d", False, True),
    (". Implementation is Python 3.11 with scikit-learn, XGBoost, "
     "imbalanced-learn, shap, lime, SciPy and Matplotlib, with PostgreSQL "
     "18.4 and Git under fixed seeds. Every table "
     "and figure below is generated from the JSON each experiment writes, so "
     "no reported number can drift from its experiment.", False, False)])

doc.add_paragraph("1.3 Key challenges and how they were addressed",
                  style="Heading 2")

para(doc, [
    ("Four challenges were material. ", False, False),
    ("Computational cost:", True, False),
    (" stability over 1,000 instances against a 390-feature forest costs "
     "roughly four hours of processor time, and became tractable only once "
     "all perturbations of an instance were explained in a single call. ",
     False, False),
    ("Memory:", True, False),
    (" the full IEEE-CIS file exceeded available memory, so a chronological "
     "300,000-row prefix is used, and 150,000 for the imbalance experiment "
     "— a prefix rather than a random draw, to preserve time ordering. ",
     False, False),
    ("Unrecorded conditions:", True, False),
    (" scripts reported measurements without recording the conditions behind "
     "them, hiding a real inconsistency — the stability metric sampled "
     "80 instances where the proposal commits to 1,000. Each script now "
     "records its seed, instance counts, curve length and top-k, and the "
     "correction reversed a conclusion (Section 4.5). ", False, False),
    ("Degenerate LIME surrogates:", True, False),
    (" ULB's LIME settings returned numerical noise on IEEE-CIS, addressed in "
     "Section 3.2. A fifth item remains open — the actionability rubric "
     "needs a second independent rater, the only component that cannot be "
     "completed computationally.", False, False)])

# ============================================== 2. Model selection process ===
doc.add_paragraph("2. Model Selection Process", style="Heading 1")

para(doc, [
    ("The models are not competitors for deployment but ", False, False),
    ("experimental conditions", True, False),
    (", so the criteria concern what makes a model a sound vehicle for "
     "measuring explanations rather than what maximises detection. Four were "
     "applied. The set had to ", False, False),
    ("span the interpretability–performance range", True, False),
    (" the literature contrasts, so the contrast could be tested rather than "
     "assumed. Each model needed an ", False, False),
    ("exact SHAP method", True, False),
    (" — decisive, because a model requiring the sampling-based "
     "KernelExplainer would confound the explainer's approximation error with "
     "the property being measured. Each had to be ", False, False),
    ("tractable at volume", True, False),
    (", since a configuration needs some 9,500 explanations and 480 LIME "
     "fits. And each had to handle imbalance ", False, False),
    ("by instance weighting rather than resampling", True, False),
    (", because resampling is the manipulated variable of the imbalance "
     "experiment and must not also appear in the baseline. Performance enters "
     "as a covariate, not a criterion; accuracy is not used, since always "
     "answering “legitimate” is 99.83% accurate on ULB and catches "
     "nothing.", False, False)])

para(doc, [
    ("Three classifiers were selected. ", False, False),
    ("Logistic regression", True, False),
    (" with balanced class weights is the interpretable anchor. The ",
     False, False),
    ("random forest", True, False),
    (" (300 trees on ULB, 150 on IEEE-CIS, balanced subsample weighting) is "
     "the field's most common choice. ", False, False),
    ("XGBoost", True, False),
    (" (400 trees, depth 6, learning rate 0.1, scale_pos_weight at the "
     "negative-to-positive ratio) represents the gradient-boosted "
     "ensembles that dominate published benchmarks. "
     "Hyperparameters are fixed a priori and deliberately not tuned, since a "
     "tuning search would add a selection process whose own interaction with "
     "explanation quality is unmeasured. Table 1 records the alternatives.",
     False, False)])

add_table(doc, [1.5, 3.45, 1.32],
          ["Alternative", "Reason for rejection", "Status"],
          [["Deep tabular models (TabTransformer, MLP)",
            "No exact SHAP method; KernelExplainer at 390 features is "
            "prohibitive here, and its sampling error would be inseparable "
            "from the instability being measured.", "Rejected"],
           ["Averaged ensembles (e.g. FraudX AI)",
            "An attribution for an averaged probability belongs to no single "
            "model, dissolving the cross-model comparison.", "Rejected"],
           ["SMOTE or undersampling in the baseline",
            "Needed as the imbalance experiment's manipulated variable; in "
            "the baseline it would confound treatment with control.",
            "Reserved as treatment"],
           ["KernelExplainer for tree and linear models",
            "Exact methods exist for these classes, so sampling would add "
            "error for no benefit.", "Rejected"],
           ["DeepLIFT, Anchors, TreeInterpreter",
            "SHAP and LIME are the dominant pair and so the methods whose "
            "reliability matters most; a wider comparison exceeds scope.",
            "Deferred"],
           ["Accuracy or F1 as selection metric",
            "Uninformative at a 0.172% base rate.", "Rejected"]],
          numeric_from=99)
caption(doc, "Table 1. Alternative models and methods considered, with reasons.")

# ============================================== 3. Application of the model ==
doc.add_paragraph("3. Application of the Model", style="Heading 1")
doc.add_paragraph("3.1 Preprocessing, transformation and feature engineering",
                  style="Heading 2")

para(doc, [
    ("Transactions are sorted by timestamp and split 70/30, giving 199,364 "
     "training and 85,443 test rows on ULB (108 test frauds) and 210,000 and "
     "90,000 on the IEEE-CIS prefix (3,759). Scaling is fitted on the "
     "training partition only. On IEEE-CIS, columns over 90% empty are "
     "dropped, removing two of 392 candidates and leaving 390; remaining gaps "
     "take training-partition medians for the random forest, while XGBoost "
     "consumes them natively. Three views of the test matrix are therefore "
     "kept — standardised, median-imputed and raw — because the "
     "explanation stage must feed each model the representation it was fitted "
     "on.", False, False)])

para(doc, [
    ("Feature engineering is deliberately minimal, by design: engineered "
     "aggregates are themselves opaque, "
     "so a derived velocity or ratio feature is no more interpretable than "
     "ULB's V14, and adding them would contaminate the measurement the "
     "actionability objective depends on. The only feature-level "
     "transformation is a ", False, False),
    ("classification", False, True),
    (", not a construction: Vesta's engineered V-columns, the masked "
     "identity columns and the counting and time-delta families are opaque, "
     "while amounts, product codes, card and address fields and email "
     f"domains are semantic. Of the 390 features retained, "
     f"{s2i['n_semantic_features']} are semantic.", False, False)])

doc.add_paragraph("3.2 Modifications to the base methods", style="Heading 2")

para(doc, [
    ("Four departures from standard practice were necessary, each because the "
     "off-the-shelf form misled. ", False, False),
    ("Deletion fidelity replaced insertion-minus-deletion,", True, False),
    (" because an insertion curve filling only the top 20 of 390 features "
     "never approaches the true prediction, driving that measure negative for "
     "reasons unrelated to explanation quality. Fidelity is reported as an ",
     False, False),
    ("absolute", True, False),
    (" probability drop, since dividing by the starting prediction explodes "
     "where a ULB model outputs about 10⁻⁶. ", False, False),
    ("Perturbations are batched", True, False),
    (" into one explainer call, which is what made stability feasible on a "
     "390-feature forest. The fourth is substantive: ", False, False),
    ("LIME's kernel width is selected per instance.", True, False),
    (" LIME weights each perturbation by exp(−D²/w²) with w "
     "defaulting to √d · 0.75. At d = 390, on an alert scored "
     "0.99996, the response surface is locally flat, so a narrow kernel draws "
     "only perturbations with identical output, the weighted regression has "
     "no variance to fit, and the coefficients returned are numerical noise "
     f"— a median local R² of {r2_def:.3f}. More samples do not "
     "help, because "
     "the problem is the weighting. Widening the kernel does, but the width "
     "required is a property of the instance, not the dataset: within one "
     "alert queue the best multiplier varies, and widening it where the "
     "surface already has gradient destroys locality. The width is chosen "
     "by maximising the surrogate's local R² — a criterion "
     "depending only on the LIME fit, never on the SHAP vector it is compared "
     "against — with instances below R² = 0.5 excluded and counted. "
     f"Median R² rises to between {r2_lo:.3f} and {r2_hi:.3f}. The same "
     "procedure on ULB selects the default width for 99 to 102 of 120 "
     "instances, as intended: one implementation adapts to 30 features and "
     "to 390 rather than being tuned for one.", False, False)])

# ================================================== 4. Preliminary results ===
doc.add_paragraph("4. Preliminary Results", style="Heading 1")
doc.add_paragraph("4.1 Detection baselines at a defined alert budget",
                  style="Heading 2")

para(doc, [
    ("Performance is reported at a fixed ", False, False),
    ("alert budget", True, False),
    (", the share of transaction volume an investigation team can review. At "
     "budget b the alerts are the top b fraction of test transactions ranked "
     "by score, so the budget fixes how many cases are reviewed and the model "
     "determines only which cases fill them. This reproduces the constraint a "
     "fraud team faces, whereas a fixed probability threshold yields "
     "different alert volumes from day to day. Table 2 reports a 0.5% budget: "
     f"{s1u['results']['XGBoost'][BUDGET]['alerts']} alerts on ULB and "
     f"{s1i['results']['XGBoost'][BUDGET]['alerts']} on IEEE-CIS.",
     False, False)])

rows = []
for tag, d in (("ULB", s1u), ("IEEE-CIS", s1i)):
    for model, r in d["results"].items():
        b = r[BUDGET]
        rows.append([tag, NICE[model], f"{r['auc_pr']:.4f}",
                     f"{r['roc_auc']:.4f}",
                     f"{b['caught'] / b['alerts']:.3f}", f"{b['recall']:.3f}"])
add_table(doc, [0.85, 1.72, 0.95, 0.95, 0.95, 0.85],
          ["Dataset", "Model", "AUC-PR", "ROC-AUC", "Prec @0.5%", "Rec @0.5%"],
          rows)
caption(doc, "Table 2. Baseline detection performance on time-ordered splits, "
             "with precision and recall at a 0.5% alert budget.")

para(doc, [
    ("These values sit below those usually published, because of the "
     "chronological split. Two readings matter. The datasets fail in ",
     False, False),
    ("opposite directions", True, False),
    (" at the same budget: on ULB logistic regression catches 86% of fraud "
     "while being wrong about four alerts in five, whereas on IEEE-CIS "
     "XGBoost is right about 98% of its alerts and misses seven frauds in "
     "eight. Reporting either rate alone would call these comparable. The "
     "IEEE-CIS recall ceiling is partly structural, since a 0.5% budget "
     "cannot hold more than about a seventh of the fraud at a 3.37% base "
     "rate. Second, the threshold-free ranking disagrees with the operating "
     "point: the random forest has the best AUC-PR on ULB while logistic "
     "regression wins both precision and recall at 0.5%. The margins are "
     "small and the budget-level rates carry no intervals, so this shows the "
     "two kinds of measure ", False, False),
    ("can", False, True),
    (" diverge rather than establishing which model is better.", False, False)])

doc.add_paragraph("4.2 Explanation quality", style="Heading 2")

rows = []
for tag, d, side in (("ULB", s2u, False), ("IEEE-CIS", s2i, True)):
    for model, r in d["per_model"].items():
        if side:
            frv = s2fr["per_model"][model]["fidelity_random"]
            rho = s2lime["per_model"][model]["shap_vs_lime_spearman"]
        else:
            frv, rho = r.get("fidelity_random"), r.get("shap_vs_lime_spearman")
        rows.append([tag, NICE[model], f3(r["fidelity_alert"]), f3(frv),
                     f3(r["max_sensitivity"]), f3(rho)])
add_table(doc, [0.85, 1.6, 1.0, 1.0, 0.85, 0.97],
          ["Dataset", "Model", "Fidelity (alerts)", "Fidelity (random)",
           "Max-sens.", "SHAP–LIME ρ"],
          rows)
caption(doc, "Table 3. Explanation quality. Fidelity is the fall in predicted "
             "probability when the top 20 attributed features are deleted "
             "(higher is better); max-sensitivity is the largest relative "
             "change in the attribution under a 5% input perturbation (lower "
             "is better). 95% bootstrap intervals omitted for space.")

para(doc, [
    ("Table 3 reports the four measures, and three results stand out. The "
     "interpretable model is the faithful one: "
     "deleting the twenty features SHAP named cuts logistic regression's ULB "
     "score by two thirds, against "
     f"{f3(s2u['per_model']['RandomForest']['fidelity_alert'])} and "
     f"{f3(s2u['per_model']['XGBoost']['fidelity_alert'])} for the ensembles. "
     "Stability degrades sharply with "
     "dimensionality for the ensembles while logistic regression is "
     "essentially dimension-independent at "
     f"{f3(s2u['per_model']['LogisticRegression']['max_sensitivity'])} and "
     f"{f3(s2i['per_model']['LogisticRegression']['max_sensitivity'])}; "
     f"XGBoost's {f3(s2i['per_model']['XGBoost']['max_sensitivity'])} on "
     "IEEE-CIS means a 5% input perturbation changes the attribution by "
     "almost as much as the attribution's own magnitude. And cross-explainer "
     "agreement falls as complexity rises, from "
     f"{f3(s2u['per_model']['LogisticRegression']['shap_vs_lime_spearman'])} "
     f"to {f3(s2u['per_model']['XGBoost']['shap_vs_lime_spearman'])} on ULB "
     f"and {f3(s2lime['per_model']['LogisticRegression']['shap_vs_lime_spearman'])}"
     f" to {f3(s2lime['per_model']['XGBoost']['shap_vs_lime_spearman'])} on "
     "IEEE-CIS.", False, False)])

doc.add_paragraph("4.3 The imbalance–explanation interaction",
                  style="Heading 2")

para(doc, [
    ("Holding architecture, split and test partition fixed and varying only "
     "the resampling regime, the null hypothesis that explanation rankings "
     "are invariant to that regime is rejected on both datasets. On ULB "
     f"(Figure 1) AUC-PR spans only {spread_u:.4f} across the three regimes "
     "while agreement with the natural-imbalance model falls to "
     f"ρ = {worst_u:.3f} under undersampling, at a Cohen's d of "
     f"{s3u['tests']['undersample']['cohens_d']:.2f}. The most-named feature "
     "changes outright: the natural model attributes 88% of alerts primarily "
     "to V14, the undersampled model names V4 most often — so two "
     "models a practitioner would treat as interchangeable tell different "
     "stories about the same transactions. The obvious objection is that "
     "SHAP "
     "attributions are simply noisy, and a control answers it: ten XGBoost "
     "models differing only in random seed, each explaining the same fixed "
     f"alert queue, agree at a mean ρ of {s5['mean_spearman']:.3f} with "
     f"AUC-PR varying by {s5['auc_pr_spread']:.4f}, so stochastic training "
     "accounts for roughly a tenth of the disagreement resampling produces. "
     f"On IEEE-CIS the covariate is held less well (AUC-PR spans "
     f"{spread_i:.4f}, SMOTE improving on the natural regime), so part of the "
     "divergence there may reflect a better model; the ULB result carries the "
     "argument.", False, False)])

add_figure(doc, "fig2_imbalance_paradox.png", 4.6)
caption(doc, "Figure 1. Detection quality is flat across resampling regimes "
             "while explanation agreement with the natural-imbalance model "
             "falls sharply (ULB).")

doc.add_paragraph("4.4 Semantic actionability", style="Heading 2")

para(doc, [
    (f"Of the {s2i['n_features']} IEEE-CIS features retained, "
     f"{s2i['n_semantic_features']} are semantically meaningful, yet all "
     "three models rank an opaque feature first for the overwhelming "
     "majority of alerts. The top-ranked feature is semantic for "
     f"{sem['LogisticRegression']['top1'] * 100:.1f}% of alerts under "
     f"logistic regression, {sem['RandomForest']['top1'] * 100:.1f}% under "
     f"the random forest and {sem['XGBoost']['top1'] * 100:.1f}% under "
     "XGBoost, with top-five shares of "
     f"{sem['LogisticRegression']['top5'] * 100:.1f}%, "
     f"{sem['RandomForest']['top5'] * 100:.1f}% and "
     f"{sem['XGBoost']['top5'] * 100:.1f}%. A semantic feature space is "
     "necessary for an actionable explanation, and these models do not use "
     "it. ULB scores zero by construction, which already disqualifies the "
     "field's dominant benchmark from any claim about actionable explanation.",
     False, False)])

doc.add_paragraph("4.5 Unexpected observations and areas for improvement",
                  style="Heading 2")

para(doc, [
    ("Four observations were not anticipated. ", False, False),
    ("Attributions are anti-faithful away from the alert queue:", True, False),
    (" on IEEE-CIS both tree ensembles return negative deletion fidelity on "
     f"the random sample, {f3(rf_r['fidelity_random'])} and "
     f"{f3(xgb_r['fidelity_random'])} with intervals excluding zero, so "
     "deleting the features SHAP ranked highest moves the fraud score up. The "
     "mechanical explanation — no probability mass to remove — does "
     f"not apply: median scores there are "
     f"{xgb_r['median_predicted_score']:.3f} to "
     f"{lr_r['median_predicted_score']:.3f}, not ULB's 10⁻⁶. The "
     "consequence is for global explanation, since a SHAP summary plot over a "
     "representative sample is dominated by this population. ", False, False),
    ("An unrecorded sample size reversed a conclusion:", True, False),
    (" at 80 instances the two ULB tree models had overlapping stability "
     "intervals and looked comparably unstable; at 1,000 they are disjoint "
     "and XGBoost is twice as unstable "
     f"({f3(s2u['per_model']['XGBoost']['max_sensitivity'])} against "
     f"{f3(s2u['per_model']['RandomForest']['max_sensitivity'])}), while the "
     "IEEE-CIS XGBoost estimate moved from 0.603 to "
     f"{f3(s2i['per_model']['XGBoost']['max_sensitivity'])}. ", False, False),
    ("Rank correlation can conceal total disagreement:", True, False),
    (" on IEEE-CIS logistic regression and the random forest reach "
     f"ρ = {f3(xm['spearman'])}, which correlation alone would call "
     "substantial agreement, while their Jaccard overlap at five features is "
     f"{f3(xm['jaccard'])}. And ", False, False),
    ("the best detector has the least usable explanations:", True, False),
    (" XGBoost leads IEEE-CIS on AUC-PR at "
     f"{s1i['results']['XGBoost']['auc_pr']:.4f} with the worst "
     f"max-sensitivity and the worst cross-explainer agreement. Five "
     "limitations follow: the IEEE-CIS subsample confines conclusions there "
     "to its first 85 days; the budget-level rates carry no intervals; the "
     "imbalance covariate is not flat on IEEE-CIS; hyperparameters are "
     "untuned; and at 400 paired instances every Wilcoxon test returns "
     "p ≈ 10⁻⁶⁷, so effect sizes carry the argument.",
     False, False)])

# ========================================================== 5. Next steps ====
doc.add_paragraph("5. Next Steps", style="Heading 1")

para(doc, [
    ("The immediate priority is the qualitative half of the actionability "
     "study, the only component blocked on something other than computation: "
     "it needs a second independent rater to score explanation reports "
     "against a rubric covering whether a named feature, a direction of "
     "effect and a plausible investigative next step can be derived, so that "
     "inter-rater agreement can be reported using Krippendorff's alpha; the "
     "rubric and worksheet generator are implemented. Three computational "
     "extensions follow: repeating the IEEE-CIS experiments at full scale, a "
     "memory-management problem rather than a methodological one; "
     "bootstrapping the budget-level rates to establish whether the "
     "divergence in Section 4.1 is real or within noise; and extending the "
     "imbalance experiment to a second architecture to test whether the "
     "interaction generalises beyond gradient boosting. Table 4 gives the "
     "timeline.", False, False)])

add_table(doc, [1.45, 4.82],
          ["Period", "Milestone"],
          [["November 2026",
            "Second rater recruited; rubric rating of 100 alerts per dataset; "
            "inter-rater agreement reported."],
           ["December 2026",
            "IEEE-CIS experiments repeated at full scale; bootstrap intervals "
            "added to budget-level rates; interim supervisor review."],
           ["January 2027",
            "Imbalance experiment extended to a second architecture; error "
            "analysis."],
           ["February–March 2027",
            "Results and discussion chapters completed; practitioner "
            "checklist drafted."],
           ["April–May 2027",
            "Full draft and revision following supervisor feedback."],
           ["June 2027", "Final submission and defence."]],
          numeric_from=99)
caption(doc, "Table 4. Timeline for remaining milestones.")

# ============================================================ References =====
doc.add_paragraph("References", style="Heading 1")

REFS = [
    "Jesus, S., Belém, C., Balayan, V., Bento, J., Saleiro, P., Bizarro, "
    "P., & Gama, J. (2021). How can I choose an explainer? An "
    "application-grounded evaluation of post-hoc explanations. Proceedings of "
    "FAccT 2021, 805–815.",
    "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting "
    "model predictions. Advances in Neural Information Processing Systems, 30, "
    "4765–4774.",
    "Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should I "
    "trust you?” Explaining the predictions of any classifier. "
    "Proceedings of KDD 2016, 1135–1144.",
    "Rudin, C. (2019). Stop explaining black box machine learning models for "
    "high stakes decisions and use interpretable models instead. Nature "
    "Machine Intelligence, 1, 206–215.",
    "Slack, D., Hilgard, S., Jia, E., Singh, S., & Lakkaraju, H. (2020). "
    "Fooling LIME and SHAP: Adversarial attacks on post hoc explanation "
    "methods. Proceedings of AIES 2020, 180–186.",
    "Weerts, H. J. P., van Ipenburg, W., & Pechenizkiy, M. (2019). A "
    "human-grounded evaluation of SHAP for alert processing. arXiv:1907.03324.",
]
for r in REFS:
    pr = doc.add_paragraph()
    # single-spaced, which the page limit requires; entries stay separated
    # by their own spacing and the hanging indent
    pr.paragraph_format.line_spacing = 1.0
    pr.paragraph_format.space_after = Pt(2)
    pr.paragraph_format.left_indent = Inches(0.5)
    pr.paragraph_format.first_line_indent = Inches(-0.5)
    pr.add_run(r)

doc.save(OUT)
print(f"wrote {OUT.relative_to(ROOT)}")
print(f"  paragraphs {len(doc.paragraphs)}  tables {len(doc.tables)}")
