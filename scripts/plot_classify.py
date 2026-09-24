# -*- coding: utf-8 -*-
"""Figure 15: classification (label-cliff) coverage failure — B6 v2 data.

Contract:
  core conclusion : set-valued CP also collapses on label cliffs in
                    classification (+35.7 to +88.0 pp across 5 MoleculeNet
                    benchmarks); conditioning on the predicted class is a
                    no-op, and APS reaches ~1.0 coverage only by a degenerate
                    set size ~2 ("cover everything").
  evidence chain  : A (hero) paired per-seed cliff vs non-cliff coverage
                    (global CP); B conditioning arms change nothing
                    (predclass == global; nnsim <= +5.4 pp); C the APS
                    trade-off — coverage bought with total set size.
  archetype       : quantitative grid (3 equal panels, double column).
  export          : figures/fig15_classify.{png,pdf}, 600 dpi PNG,
                    editable PDF text, alignment gate before export.
Data: results_server/B6V2_METHODS.csv (5 datasets x 10 seeds x 4 methods).
Panel-A gap tags and direction-consistency counts sit BELOW the axis (a
stroke-free strip) and the colour key sits ABOVE the axes, so no text is
crossed by the per-seed spaghetti. Panel-C x-range contains every mean
(global-CP sets may be empty on cliffs -> mean size can be < 1).
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SKILL_SCRIPTS = r"C:/Users/Administrator/.workbuddy/skills/nature-figure/scripts"
if SKILL_SCRIPTS not in sys.path:
    sys.path.insert(0, SKILL_SCRIPTS)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

RED = "#c0392b"     # cliff / harm
GRAY = "#7f8c8d"    # noncliff / neutral
BLUE = "#2c7fb8"    # APS accent
DARK = "#2c3e50"

D = pd.read_csv("results_server/B6V2_METHODS.csv")
DATASETS = ["BACE", "BBBP", "ClinTox", "HIV", "Tox21_NR-AR"]
order = DATASETS
TICK = {"BACE": "BACE", "BBBP": "BBBP", "ClinTox": "ClinTox",
        "HIV": "HIV", "Tox21_NR-AR": "Tox21"}  # full name in the caption

g = D[D.method == "global"].copy()
g["per_gap_pp"] = 100 * (g.cov_noncliff - g.cov_cliff)  # same convention as
agg = g.groupby("dataset").agg(                         # the server SUMMARY.txt
    cliff=("cov_cliff", "mean"), non=("cov_noncliff", "mean"),
    n_cliff=("n_cliff", "mean"), size=("size_cliff", "mean"),
    gap_pp=("per_gap_pp", "mean"), pos=("per_gap_pp", lambda s: int((s > 0).sum())),
    nseeds=("per_gap_pp", "size"))

aps = D[D.method == "aps"].groupby("dataset").agg(
    cliff=("cov_cliff", "mean"), size=("size_cliff", "mean"))
nnsim = D[D.method == "mondrian_nnsim"].groupby("dataset")["cov_cliff"].mean()
pred = D[D.method == "mondrian_predclass"].groupby("dataset")["cov_cliff"].mean()

fig = plt.figure(figsize=(7.1, 2.85))
gs = fig.add_gridspec(1, 3, left=0.065, right=0.988, top=0.83, bottom=0.265,
                      wspace=0.34)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])
xtr = axA.get_xaxis_transform()  # x in data coords, y in axes fraction

# ---- Panel A: paired per-seed cliff vs non-cliff coverage (hero) ----
axA.axhline(0.90, color="#999999", lw=0.8, ls=(0, (4, 3)), zorder=1)
axA.text(0.99, 1.055, "nominal 0.90", transform=axA.transAxes,
         fontsize=6.5, color="#777777", ha="right", va="center")
for i, ds in enumerate(order):
    sub = g[g.dataset == ds]
    xs_c, xs_n = i - 0.16, i + 0.16
    for _, r in sub.iterrows():  # one grey spaghetti per seed
        axA.plot([xs_c, xs_n], [r.cov_cliff, r.cov_noncliff],
                 color="#d5d8dc", lw=0.7, zorder=2)
    mc, mn = agg.loc[ds, "cliff"], agg.loc[ds, "non"]
    axA.plot([xs_c], [mc], "o", color=RED, ms=4.5, zorder=4)
    axA.plot([xs_n], [mn], "o", markerfacecolor="white",
             markeredgecolor=GRAY, ms=4.5, zorder=4)
    # gap + direction consistency live below the axis: a stroke-free strip
    axA.text(i, -0.15, f"+{agg.loc[ds,'gap_pp']:.1f}", transform=xtr,
             ha="center", va="top", fontsize=6.4, color=DARK,
             fontweight="bold")
    axA.text(i, -0.30, f"{int(agg.loc[ds,'pos'])}/{int(agg.loc[ds,'nseeds'])}",
             transform=xtr, ha="center", va="top", fontsize=6.0,
             color="#888888")
# colour key in the strip ABOVE the axes (stroke-free: the spaghetti is
# clipped at the axes, and some seeds reach coverage ~1.0 inside the panel,
# so the key must not sit at data y ~ 1.0)
keyy = 1.055
axA.text(-0.02, keyy, "\u25cf", color=RED, fontsize=7, ha="center",
         va="center", transform=axA.transAxes)
axA.text(0.04, keyy, "cliff", color=DARK, fontsize=6.6, ha="left",
         va="center", transform=axA.transAxes)
axA.text(0.24, keyy, "\u25cb", color=GRAY, fontsize=7, ha="center",
         va="center", transform=axA.transAxes)
axA.text(0.30, keyy, "non-cliff", color=DARK, fontsize=6.6, ha="left",
         va="center", transform=axA.transAxes)
axA.set_xticks(range(len(order)))
axA.set_xticklabels([TICK[d] for d in order])
axA.tick_params(axis="y", pad=5)
axA.set_ylim(-0.02, 1.06)
axA.set_xlim(-0.55, 4.55)
axA.set_ylabel("coverage (\u03b1 = 0.10)")
axA.text(-0.30, 1.055, "a", transform=axA.transAxes, fontsize=13,
         fontweight="bold", va="bottom", ha="left")

# ---- Panel B: conditioning arms do not repair ----
xs = np.arange(len(order))
d_nns = np.array([(nnsim[d] - agg.loc[d, "cliff"]) * 100 for d in order])
d_pre = np.array([(pred[d] - agg.loc[d, "cliff"]) * 100 for d in order])
axB.axhline(0, color="#999999", lw=0.8)
axB.bar(xs - 0.17, d_pre, width=0.30, color="#b0bec5", zorder=2,
        label="\u2190 predicted class")
axB.bar(xs + 0.17, d_nns, width=0.30, color=BLUE, zorder=2,
        label="\u2190 NN similarity")
for x, v in zip(xs + 0.17, d_nns):
    if v > 0.05:
        axB.annotate(f"+{v:.1f}", xy=(x, v), xytext=(0, 2),
                     textcoords="offset points", ha="center", fontsize=6.5,
                     color=DARK)
axB.set_xticks(xs)
axB.set_xticklabels([TICK[d] for d in order])
axB.tick_params(axis="y", pad=5)
axB.set_ylabel("\u0394 label-cliff coverage\nvs global (pp)", labelpad=4)
axB.set_ylim(-1.2, 7.4)
axB.legend(loc="upper left", handletextpad=0.4, borderaxespad=0.2)
axB.text(-0.28, 1.055, "b", transform=axB.transAxes, fontsize=13,
         fontweight="bold", va="bottom", ha="left")

# ---- Panel C: APS is degenerate ----
for ds in order:
    sub = D[(D.dataset == ds) & (D.method.isin(["global", "aps"]))]
    for m, col, mk in [("global", GRAY, "o"), ("aps", BLUE, "s")]:
        s2 = sub[sub.method == m]
        axC.scatter(s2.size_cliff, s2.cov_cliff, s=9, color=col, alpha=0.28,
                    marker=mk, linewidths=0, zorder=2)
gm = agg.loc[order, "size"]
axC.scatter(gm, agg.loc[order, "cliff"], s=26, color=RED, zorder=4,
            label="Global CP", marker="o")
axC.scatter(aps.loc[order, "size"], aps.loc[order, "cliff"], s=26,
            facecolors="white", edgecolors=BLUE, zorder=4,
            label="APS", marker="s")
axC.axhline(0.90, color="#999999", lw=0.8, ls=(0, (4, 3)), zorder=1)
# NB global-CP sets may be empty on cliff compounds (conformal_sets does not
# force the argmax in), so the mean size can fall below 1 and even to 0;
# the x-range must contain those means or clipped-out marker geometry
# re-surfaces in the PDF vector stream on top of the tick labels.
axC.set_xlabel("mean label-cliff set size (labels)")
axC.set_ylabel("label-cliff coverage")
axC.set_xlim(-0.05, 2.20)
axC.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
axC.set_ylim(-0.02, 1.06)
axC.tick_params(axis="y", pad=5)
axC.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.2)
axC.set_title("APS: both labels in every set", fontsize=7, pad=4,
              color=DARK)
axC.text(-0.24, 1.055, "c", transform=axC.transAxes, fontsize=13,
         fontweight="bold", va="bottom", ha="left")

require_matplotlib_panel_alignment(
    fig,
    json_out="figures/fig15_classify.alignment.json",
    overlay_svg="figures/fig15_classify.alignment.svg",
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    strict=True,
)

fig.savefig("figures/fig15_classify.png", bbox_inches="tight")
fig.savefig("figures/fig15_classify.pdf", bbox_inches="tight")
print("saved: figures/fig15_classify.png + .pdf | "
      f"gaps: {[round(agg.loc[d,'gap_pp'],1) for d in order]}")
