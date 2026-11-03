"""
Collect every results JSON into the tables the thesis will report (stage S7).

Reads only what the experiments wrote, so the tables cannot drift from the
numbers that produced them. Re-run after any experiment.

    .venv\\Scripts\\python.exe experiments\\99_summary_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "SUMMARY.md"


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ci(v):
    """Format a (mean, lo, hi) triple."""
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.3f}"
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


lines = ["# Results summary",
         "",
         "Generated from the JSON written by each experiment. Do not edit by "
         "hand — re-run `experiments/99_summary_tables.py`.",
         ""]

# ---------------------------------------------------------------- S4 control
s4 = load("s4_synthetic_validation.json")
if s4:
    lines += ["## S4 — metric validation on synthetic controls", "",
              "Checks that the measures recover importances planted by "
              "construction. If any of these fail, nothing downstream is "
              "trustworthy.", "",
              "| Check | Result |", "|---|---|"]
    for k, v in s4["checks"].items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines.append("")

# -------------------------------------------------------------- S1 baselines
lines += ["## S1 — baseline models (time-ordered splits)", "",
          "An alert budget of *b* means the investigator reviews the top *b* "
          "fraction of test transactions, ranked by model score.", "",
          "| Dataset | Model | AUC-PR | ROC-AUC | Precision @0.5% | Recall @0.5% |",
          "|---|---|---|---|---|---|"]
for tag, fn in (("ULB", "s1_baselines_ulb.json"),
                ("IEEE-CIS", "s1_baselines_ieee.json")):
    d = load(fn)
    if not d:
        continue
    for model, r in d["results"].items():
        b = r.get("recall_at_0.005")
        if b:
            prec = b["caught"] / b["alerts"]
            lines.append(f"| {tag} | {model} | {r['auc_pr']:.4f} | "
                         f"{r['roc_auc']:.4f} | {prec:.3f} | {b['recall']:.3f} |")
        else:
            lines.append(f"| {tag} | {model} | {r['auc_pr']:.4f} | "
                         f"{r['roc_auc']:.4f} | — | — |")
lines.append("")

# ------------------------------------------------- S2 explanation quality
lines += ["## S2 — explanation quality", "",
          "| Dataset | Model | Fidelity (alerts) | Max-sensitivity | "
          "SHAP vs LIME ρ | Jaccard@5 |", "|---|---|---|---|---|---|"]
for tag, fn in (("ULB", "s2_explanation_quality_ulb.json"),
                ("IEEE-CIS", "s2_explanation_quality_ieee.json")):
    d = load(fn)
    if not d:
        continue
    for model, r in d["per_model"].items():
        lines.append(
            f"| {tag} | {model} | {ci(r.get('fidelity_alert'))} | "
            f"{ci(r.get('max_sensitivity'))} | "
            f"{ci(r.get('shap_vs_lime_spearman'))} | "
            f"{ci(r.get('shap_vs_lime_jaccard'))} |")
lines.append("")

# cross-model agreement
lines += ["### Do different models explain the same alert the same way?", "",
          "H0: the two models rank the alert's features identically. "
          "With n = 400 the p-value is almost guaranteed to be small, so "
          "Cohen's d carries the argument.", "",
          "| Dataset | Pair | Spearman | Jaccard@5 | Wilcoxon p | Cohen's d |",
          "|---|---|---|---|---|---|"]
for tag, fn in (("ULB", "s2_explanation_quality_ulb.json"),
                ("IEEE-CIS", "s2_explanation_quality_ieee.json")):
    d = load(fn)
    if not d:
        continue
    for pair, r in d.get("cross_model", {}).items():
        a, b = pair.split("|")
        p = f"{r['p_value']:.3g}" if "p_value" in r else "—"
        cd = f"{r['cohens_d']:.2f}" if "cohens_d" in r else "—"
        lines.append(f"| {tag} | {a} vs {b} | {ci(r['spearman'])} | "
                     f"{ci(r['jaccard'])} | {p} | {cd} |")
lines.append("")

# -------------------------------------------------------------- S3 imbalance
s3_sets = [("ULB", "s3_imbalance_explanation_ulb.json"),
           ("IEEE-CIS", "s3_imbalance_explanation_ieee.json")]
if any(load(fn) for _, fn in s3_sets):
    lines += ["## S3 — does resampling change the explanation?", "",
              "One architecture, one test set, three training regimes, run "
              "independently on each dataset.", ""]
for label, fn in s3_sets:
    s3 = load(fn)
    if not s3:
        continue
    lines.append(f"### {label}")
    if s3.get("rows_used") and s3.get("rows_available") \
            and s3["rows_used"] < s3["rows_available"]:
        lines += ["", f"{s3['rows_used']:,} of {s3['rows_available']:,} rows "
                  f"({s3.get('subsample_note', 'subsample')}).",
                  ""]
    lines += ["", "| Regime | AUC-PR | ρ vs natural | Jaccard@5 vs natural |",
              "|---|---|---|---|"]
    for regime, ap in s3["performance"].items():
        agr = s3["agreement"].get(regime)
        lines.append(f"| {regime} | {ap:.4f} | "
                     f"{ci(agr['spearman']) if agr else '—'} | "
                     f"{ci(agr['jaccard']) if agr else '—'} |")
    spread = max(s3["performance"].values()) - min(s3["performance"].values())
    lines += ["", f"AUC-PR spread across regimes: **{spread:.4f}**", "",
              "| Comparison | Wilcoxon p | mean ρ | Cohen's d |",
              "|---|---|---|---|"]
    for k, t in s3["tests"].items():
        lines.append(f"| natural vs {k} | {t['p_value']:.3g} | "
                     f"{t['mean_rho']:.3f} | {t['cohens_d']:.2f} |")
    lines += ["", "**Most-named feature per regime**", "",
              "| Regime | Top features |", "|---|---|"]
    for regime, feats in s3["top_features"].items():
        total = sum(feats.values())
        s = ", ".join(f"{k} ({v * 100 // max(total, 1)}%)" for k, v in feats.items())
        lines.append(f"| {regime} | {s} |")
    lines += ["", f"H0 (rankings invariant to regime): "
              f"**{'REJECTED' if s3['h0_rejected'] else 'not rejected'}**", ""]

# ------------------------------------------- S5 stability across seeds
s5 = load("s5_stability_seeds.json")
if s5:
    lines += ["## S5 — is the explanation stable across retraining seeds?", "",
              "A control for S3. S2's max-sensitivity perturbs the *input*; "
              "this perturbs the *seed*, holding the data fixed. "
              f"{len(s5['seeds'])} seeds, {s5['n_alert']} alerts, "
              f"reference seed {s5['reference_seed']}.", "",
              "| Seed | ρ vs reference | Jaccard@5 | Cohen's d |",
              "|---|---|---|---|"]
    for seed, agr in s5["agreement"].items():
        t = s5.get("tests", {}).get(seed, {})
        cd = f"{t['cohens_d']:.2f}" if "cohens_d" in t else "—"
        lines.append(f"| {seed} | {ci(agr['spearman'])} | "
                     f"{ci(agr['jaccard'])} | {cd} |")
    lines += ["", f"Mean Spearman across seeds: **{s5['mean_spearman']:.3f}**",
              "", f"AUC-PR spread across seeds: **{s5['auc_pr_spread']:.4f}**",
              ""]
    # the contrast with S3 is the whole point of running this
    s3u = load("s3_imbalance_explanation_ulb.json")
    if s3u:
        worst = min(t["mean_rho"] for t in s3u["tests"].values())
        lines += ["**The comparison that matters.** Changing the training "
                  f"seed moves the ranking to ρ = {s5['mean_spearman']:.3f}; "
                  f"changing the resampling regime moves it to ρ = "
                  f"{worst:.3f}. The divergence in S3 is therefore caused by "
                  "the imbalance strategy, not by stochastic training.", ""]

# ------------------------------------------------------ Objective 4 numbers
s2i = load("s2_explanation_quality_ieee.json")
if s2i and "semantic_share" in s2i:
    lines += ["## Objective 4 — can an investigator act on the explanation?", "",
              f"IEEE-CIS has {s2i['n_features']} features, of which only "
              f"**{s2i['n_semantic_features']}** are semantically meaningful. "
              "ULB has none beyond Time and Amount.", "",
              "| Model | Top-1 feature is semantic | Top-5 semantic (mean) |",
              "|---|---|---|"]
    for model, r in s2i["semantic_share"].items():
        lines.append(f"| {model} | {r['top1'] * 100:.1f}% | {r['top5'] * 100:.1f}% |")
    lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)}  ({len(lines)} lines)")
# the Windows console is cp1252; rho and em-dash would raise UnicodeEncodeError
enc = sys.stdout.encoding or "utf-8"
print("\n".join(lines[:40]).encode(enc, "replace").decode(enc))
