"""
Correct fonts, sizes and emphasis in the hand-edited Progress Report 5.

Run against reports/ProgressReport5-NamigPlanov.docx after editing its prose
in Word. It changes formatting only -- never a character of text -- and
asserts that at the end.

Three things it repairs.

1. Heading styles lose their font. Saving in Word reset Heading 1 and
   Heading 2 to the document theme, so they carry no font name, size or
   colour and render as Calibri Light, 14/13 pt, accent blue. The theme
   attributes (w:asciiTheme) override a plain w:ascii, so they have to be
   removed rather than overwritten.

2. Three body paragraphs carry the Heading 2 style. Indices 8, 28 and 31 are
   running prose that Word styled as headings during editing, so they render
   bold italic and appear in the navigation pane as section titles.

3. Emphasis is gone. Editing flattened every bold and italic run inside the
   body text, so defined terms, model names and statistical symbols have lost
   the formatting that distinguished them. Symbols are italicised per APA.

    .venv\\Scripts\\python.exe reports\\fix_pr5_formatting.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "reports" / "ProgressReport5-NamigPlanov.docx"

TNR = "Times New Roman"
BODY_PT = 12.0
SMALL_PT = 9.5          # tables and captions
HEADER_PT = 11.0        # the byline block under the title

# body paragraphs that Word turned into headings during editing
MISSTYLED = (8, 28, 31)

# (paragraph index, regex, bold, italic, max replacements)
# the formatting is applied to capture group 1, so a rule can italicise one
# symbol inside a longer phrase
EMPHASIS = [
    # --- defined terms -----------------------------------------------------
    (8,  r"\b(interpretability)\b",                 False, True,  1),
    (8,  r"\b(explainability)\b",                   False, True,  1),
    (8,  r"(explanation quality)",                  False, True,  1),
    # --- datasets, first mention ------------------------------------------
    (10, r"\b(ULB)\b",                              True,  False, 1),
    (10, r"\b(IEEE-CIS)\b",                         True,  False, 1),
    # --- statistical symbols (APA italicises these) -----------------------
    (11, r"Cohen's (d)\b",                          False, True,  0),
    (33, r"Cohen's (d)\b",                          False, True,  0),
    (26, r"top (b) fraction",                       False, True,  0),
    (39, r"(p) ≈",                             False, True,  0),
    (39, r"(ρ) = 0\.553",                      False, True,  0),
    # --- the claims the sections turn on ----------------------------------
    (15, r"(experimental conditions)",              True,  False, 1),
    (15, r"(exact SHAP method)",                    True,  False, 1),
    (26, r"(set alert budget)",                     True,  False, 1),
    (28, r"(opposite ends)",                        True,  False, 1),
    # --- model names, first mention ---------------------------------------
    (16, r"\b(logistic regression)\b",              True,  False, 1),
    (16, r"\b(random forest)\b",                    True,  False, 1),
    (16, r"\b(XGBoost)\b",                          True,  False, 1),
    # --- classification vs construction -----------------------------------
    (21, r"it is a (classification)",               False, True,  1),
    (21, r"not a (construction)",                   False, True,  1),
    # --- the substantive methodological change ----------------------------
    (23, r"(The width of the LIME kernel is chosen for each instance)",
                                                    True,  False, 1),
]


def force_font(style, family=TNR):
    """Pin a style to a real font family, theme attributes included.

    Word names the built-in heading fonts through the theme
    (w:asciiTheme="majorHAnsi"), and that wins over the plain w:ascii that
    python-docx writes, so the attribute has to go.
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


def spans_for(text, idx, rules):
    """Non-overlapping (start, end, bold, italic) spans for one paragraph."""
    found = []
    for p_idx, pattern, bold, italic, limit in rules:
        if p_idx != idx:
            continue
        for n, m in enumerate(re.finditer(pattern, text)):
            if limit and n >= limit:
                break
            found.append((m.start(1), m.end(1), bold, italic))
    found.sort()
    out = []
    for s in found:
        if out and s[0] < out[-1][1]:
            continue            # overlaps an earlier span; keep the first
        out.append(s)
    return out


def apply_emphasis(par, spans, size_pt):
    """Rebuild a paragraph's runs so the given spans carry bold/italic.

    The text is reassembled verbatim from the original; only run boundaries
    and their formatting change.
    """
    text = par.text
    for r in list(par.runs):
        r._element.getparent().remove(r._element)
    cursor = 0
    pieces = []
    for start, end, bold, italic in spans:
        if start > cursor:
            pieces.append((text[cursor:start], False, False))
        pieces.append((text[start:end], bold, italic))
        cursor = end
    if cursor < len(text):
        pieces.append((text[cursor:], False, False))
    for chunk, bold, italic in pieces:
        run = par.add_run(chunk)
        run.font.name = TNR
        run.font.size = Pt(size_pt)
        run.font.color.rgb = RGBColor(0, 0, 0)
        run.bold, run.italic = bold or None, italic or None
    assert par.text == text, "text changed while reformatting"


def main() -> int:
    doc = Document(DOC)
    before = [p.text for p in doc.paragraphs]

    # ---- 1. styles --------------------------------------------------------
    for name, italic in (("Normal", None), ("Heading 1", None),
                         ("Heading 2", True)):
        st = doc.styles[name]
        st.font.name = TNR
        st.font.size = Pt(BODY_PT)
        st.font.color.rgb = RGBColor(0, 0, 0)
        if name != "Normal":
            st.font.bold = True
            st.font.italic = italic
        force_font(st)
    print(f"styles pinned to {TNR} {BODY_PT:.0f} pt, black")

    # ---- 2. body paragraphs wrongly styled as headings -------------------
    for i in MISSTYLED:
        par = doc.paragraphs[i]
        was = par.style.name
        par.style = doc.styles["Normal"]
        pf = par.paragraph_format
        pf.line_spacing = 1.5
        pf.space_before = Pt(0)
        pf.space_after = Pt(2)
        pf.keep_with_next = False
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        print(f"  paragraph {i}: {was} -> Normal  ({par.text[:44]}...)")

    # ---- 3. sizes and emphasis -------------------------------------------
    captions = [i for i, p in enumerate(doc.paragraphs)
                if re.match(r"^(Table|Figure) \d+\.", p.text.strip())]
    emphasised = 0
    for i, par in enumerate(doc.paragraphs):
        if not par.text.strip():
            continue
        if i in captions:
            size = SMALL_PT
        elif 2 <= i <= 4:
            size = HEADER_PT
        else:
            size = BODY_PT
        spans = spans_for(par.text, i, EMPHASIS)
        if spans:
            apply_emphasis(par, spans, size)
            emphasised += len(spans)
        else:
            for r in par.runs:
                r.font.name = TNR
                r.font.size = Pt(size)
    print(f"sizes normalised; {emphasised} spans emphasised "
          f"across {len({r[0] for r in EMPHASIS})} paragraphs")

    # ---- table and caption runs ------------------------------------------
    cells = 0
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                for par in cell.paragraphs:
                    for r in par.runs:
                        r.font.name = TNR
                        r.font.size = Pt(SMALL_PT)
                        cells += 1
    print(f"table runs normalised to {SMALL_PT} pt: {cells}")

    after = [p.text for p in doc.paragraphs]
    assert before == after, "paragraph text changed -- refusing to save"
    doc.save(DOC)
    print(f"saved {DOC.relative_to(ROOT)} (text verified unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
