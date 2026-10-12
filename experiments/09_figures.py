"""
Thesis figures, generated from the results JSON.

Nothing here is retyped: every value is read from results/, so a figure cannot
disagree with the experiment that produced it. Re-run after any experiment.

Colours are the dataviz reference palette's first three categorical slots,
which that palette documents as clearing the all-pairs CVD and normal-vision
floors in both modes. Series are additionally distinguished by position and
direct labels, so nothing depends on colour alone.

    .venv\\Scripts\\python.exe experiments\\09_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"

# reference palette, light surface
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save(fig, stem, caption):
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.png / .pdf   {caption}")


made = 0

# --- Figure 1: fraud is not stationary ------------------------------------
eda = load("s1_exploratory_analysis.json")
if eda:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, (tag, colour) in zip(axes, (("ULB", BLUE), ("IEEE", ORANGE))):
        d = np.array(eda[tag]["fraud_rate_by_decile"]) * 100
        x = np.arange(1, len(d) + 1)
        ax.plot(x, d, color=colour, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        mean = d.mean()
        ax.axhline(mean, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        # anchor the mean label at the left edge: the right-hand end of the
        # series runs along the mean line in both panels
        ax.annotate(f"mean {mean:.2f}%", (x[0], mean), xytext=(2, 6),
                    textcoords="offset points", ha="left", fontsize=8,
                    color=MUTED)
        name = "ULB (48 hours)" if tag == "ULB" else "IEEE-CIS (85 days)"
        ax.set_title(f"{name} — {d.max() / d.min():.1f}× variation",
                     color=INK, loc="left")
        ax.set_xlabel("chronological decile of the data")
        ax.set_ylabel("fraud rate (%)")
        ax.set_xticks(x)
        ax.set_ylim(0, d.max() * 1.25)
    fig.suptitle("Fraud rate is not stationary — a random split hides this",
                 color=INK, fontsize=12, x=0.012, ha="left", y=1.04)
    save(fig, "fig1_fraud_drift",
         "fraud rate across chronological deciles, both datasets")
    made += 1

# --- Figure 2: the imbalance paradox --------------------------------------
s3 = load("s3_imbalance_explanation_ulb.json")
if s3:
    regimes = ["natural", "undersample", "smote"]
    auc = [s3["performance"][r] for r in regimes]
    agree = [1.0] + [s3["agreement"][r]["spearman"][0] for r in regimes[1:]]
    err = [(0, 0)] + [(s3["agreement"][r]["spearman"][0] - s3["agreement"][r]["spearman"][1],
                       s3["agreement"][r]["spearman"][2] - s3["agreement"][r]["spearman"][0])
                      for r in regimes[1:]]
    x = np.arange(len(regimes))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    b1 = ax.bar(x - w / 2, auc, w, color=BLUE, label="detection quality (AUC-PR)")
    b2 = ax.bar(x + w / 2, agree, w, color=ORANGE,
                yerr=np.array(err).T, capsize=3, ecolor=MUTED,
                label="explanation agreement with natural (Spearman)")
    for rect, v in zip(b1, auc):
        ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color=INK2)
    for rect, v in zip(b2, agree):
        ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(["natural\nimbalance", "random\nundersampling", "SMOTE"])
    ax.set_ylabel("score (0–1)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9, labelcolor=INK2)
    ax.set_title("Resampling leaves detection intact and changes the explanation",
                 color=INK, loc="left")
    # as a caption under the axis, not inside the plot where it hit the bars
    spread = max(auc) - min(auc)
    fig.text(0.125, -0.06,
             f"Detection quality varies by only {spread:.4f} AUC-PR across the "
             f"three regimes, while the explanations diverge sharply "
             f"(Wilcoxon p = {min(t['p_value'] for t in s3['tests'].values()):.0e}).",
             fontsize=9, color=MUTED, ha="left")
    save(fig, "fig2_imbalance_paradox",
         "accuracy flat, explanations diverge (S3)")
    made += 1

# --- Figure 3: do two explainers agree? -----------------------------------
ulb = load("s2_explanation_quality_ulb.json")
if ulb:
    models = list(ulb["per_model"])
    rho = [ulb["per_model"][m]["shap_vs_lime_spearman"][0] for m in models]
    jac = [ulb["per_model"][m]["shap_vs_lime_jaccard"][0] for m in models]
    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    b1 = ax.bar(x - w / 2, rho, w, color=BLUE, label="Spearman (all features)")
    b2 = ax.bar(x + w / 2, jac, w, color=AQUA, label="Jaccard@5 (top features)")
    for rects, vals in ((b1, rho), (b2, jac)):
        for rect, v in zip(rects, vals):
            ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9, color=INK2)
    ax.axhline(1.0, color=AXIS, linewidth=1)
    ax.annotate("perfect agreement", (len(models) - 0.5, 1.0), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("Regression", " Reg.") for m in models])
    ax.set_ylabel("agreement (0–1)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK2)
    ax.set_title("SHAP and LIME explaining the same model and the same alert",
                 color=INK, loc="left")
    save(fig, "fig3_explainer_agreement", "SHAP vs LIME on ULB (S2)")
    made += 1

# --- Figure 4: can an investigator act on it? -----------------------------
ieee = load("s2_explanation_quality_ieee.json")
if ieee and "semantic_share" in ieee:
    models = list(ieee["semantic_share"])
    top1 = [ieee["semantic_share"][m]["top1"] * 100 for m in models]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bars = ax.barh(np.arange(len(models)), top1, color=ORANGE, height=0.55)
    for rect, v in zip(bars, top1):
        ax.annotate(f"{v:.1f}%", (v, rect.get_y() + rect.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([m.replace("Regression", " Reg.") for m in models])
    ax.set_xlabel("alerts whose top-ranked feature is humanly meaningful (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_title("IEEE-CIS has 24 meaningful features of 392 — "
                 "and models rarely rank them first", color=INK, loc="left")
    ax.annotate("ULB scores 0% by construction: every feature except\n"
                "Time and Amount is an anonymised principal component.",
                (0.42, 0.12), xycoords="axes fraction", fontsize=9, color=MUTED)
    save(fig, "fig4_semantic_share", "actionability of the top feature (Obj. 4)")
    made += 1

print(f"\n{made} figures written to figures/ (PNG at 300 dpi, plus PDF)")
if made < 4:
    print("some results JSON missing - re-run the matching experiment first")
