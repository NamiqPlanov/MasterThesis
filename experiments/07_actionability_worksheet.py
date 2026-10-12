"""
Stage S6 (Secondary Objective 4): build the actionability rating instrument.

The proposal promises explanation reports for matched alerts from both
datasets, "scored against a predefined rubric covering whether a named
feature, a direction and a plausible investigative next step can be derived",
rated independently by two raters with agreement reported.

This script produces that instrument: a rubric and two worksheets, one per
dataset, each holding the top-5 SHAP features for 100 alerts in the form an
investigator would actually be shown. The rating itself is human work and is
deliberately not automated -- automating it would answer the question by
assuming it.

Raters never see which dataset a row came from while scoring: the worksheets
are also written in a single shuffled, blinded file.

    .venv\\Scripts\\python.exe experiments\\07_actionability_worksheet.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "actionability"
ART = ROOT / "artifacts"
SEED = 42
N_PER_DATASET = 100
TOP_K = 5

RUBRIC = """# Actionability rubric — AZ / XAI fraud explanations

You are a fraud investigator. For each alert below you are shown the five
features the model considered most important, with the direction and size of
each contribution. **You are not told the dataset.**

Answer three questions per alert. Write 1 for yes, 0 for no. If you genuinely
cannot tell, write `?` — do not guess.

| Code | Question |
|---|---|
| **Q1 — named** | Can you state what at least one of the listed features *means*? |
| **Q2 — direction** | Can you tell whether the top feature pushed the model *toward* fraud or *away* from it? |
| **Q3 — next step** | Based on this explanation, can you name one concrete thing you would check next? |

Guidance:

* Q1 is about meaning, not familiarity. `TransactionAmt` means something.
  `V201` does not, even after you have seen it a hundred times.
* Q3 must be a real action — "check whether this card has been used from this
  email domain before" counts; "investigate further" does not.
* Rate independently. Do not discuss rows with the other rater until both
  worksheets are complete.

Return the completed file unchanged apart from the three columns.
"""


def explain(tag, model_file, split_file, input_key, n):
    split = joblib.load(ART / split_file)
    cols = split["columns"]
    model = joblib.load(ART / model_file)
    X = split[input_key]
    if tag == "IEEE":
        X = np.where(np.isnan(X), split["median"], X).astype(np.float32)
    scores = model.predict_proba(X)[:, 1]
    idx = np.argsort(-scores)[:n]
    sv = np.asarray(shap.TreeExplainer(model).shap_values(X[idx]))
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    rows = []
    for r, i in enumerate(idx):
        top = np.argsort(-np.abs(sv[r]))[:TOP_K]
        parts = [f"{cols[j]} {'+' if sv[r][j] > 0 else '-'}{abs(sv[r][j]):.3f}"
                 for j in top]
        rows.append({"source": tag, "alert_rank": r + 1,
                     "model_score": round(float(scores[i]), 4),
                     "explanation": "  |  ".join(parts)})
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows = []
    rows += explain("ULB", "ulb_RandomForest.joblib", "ulb_split.joblib",
                    "Xte", N_PER_DATASET)
    rows += explain("IEEE", "ieee_RandomForest.joblib", "ieee_split.joblib",
                    "Xte_raw", N_PER_DATASET)

    df = pd.DataFrame(rows)
    df["item_id"] = [f"A{i:03d}" for i in range(1, len(df) + 1)]

    # blinded, shuffled worksheet: raters must not know which dataset a row is
    blind = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    blind["item_id"] = [f"B{i:03d}" for i in range(1, len(blind) + 1)]
    key = blind[["item_id", "source", "alert_rank"]]
    sheet = blind[["item_id", "model_score", "explanation"]].copy()
    for c in ("Q1_named", "Q2_direction", "Q3_next_step"):
        sheet[c] = ""

    (OUT / "RUBRIC.md").write_text(RUBRIC, encoding="utf-8")
    for rater in ("rater1", "rater2"):
        sheet.to_csv(OUT / f"worksheet_{rater}.csv", index=False,
                     encoding="utf-8-sig")
    key.to_csv(OUT / "blinding_key.csv", index=False, encoding="utf-8-sig")

    print("=" * 74)
    print("  S6 - ACTIONABILITY RATING INSTRUMENT")
    print("=" * 74)
    print(f"  alerts per dataset : {N_PER_DATASET}")
    print(f"  total items        : {len(sheet)} (shuffled and blinded)")
    print(f"  top features shown : {TOP_K}")
    print("\n  example item as a rater sees it:")
    ex = sheet.iloc[0]
    print(f"    {ex['item_id']}  score {ex['model_score']}")
    print(f"    {ex['explanation'][:150]}")
    print("\n  wrote:")
    for f in ("RUBRIC.md", "worksheet_rater1.csv", "worksheet_rater2.csv",
              "blinding_key.csv"):
        print(f"    results/actionability/{f}")
    print("\n  NEXT: two raters complete the worksheets independently,")
    print("        then 08_actionability_analysis.py computes agreement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
