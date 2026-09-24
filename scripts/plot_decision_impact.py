# -*- coding: utf-8 -*-
"""Fig 11 — 决策影响：低估风险（把真活性悬崖分子判弱而淘汰）。
数据：results_decision/DECISION_IMPACT.csv。300 dpi。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 600, "figure.dpi": 100})
RED, GRAY, GREEN, BLUE = "#c0392b", "#7f8c8d", "#1e8449", "#2c5f8a"

R = pd.read_csv("results_decision/DECISION_IMPACT.csv")

def ms(df, col):
    v = df.groupby("dataset")[col].mean()
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))

def letter(ax, ch):
    ax.text(-0.14, 1.06, ch, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="top", ha="left")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.0, 4.0),
                               gridspec_kw={"width_ratios": [1.05, 1.0]})

# ---- Panel A: under-rate vs tau (global), cliff vs non-cliff ----
taus = sorted(R["tau_q"].unique())
g = R[R.method == "global"]
mc = [ms(g[(g.tau_q == t) & (g.group == "cliff")], "under_rate") for t in taus]
mn = [ms(g[(g.tau_q == t) & (g.group == "noncliff")], "under_rate") for t in taus]
xc = np.arange(len(taus))
axA.errorbar(xc, [m for m, _ in mc], yerr=[s for _, s in mc], fmt="o-", color=RED,
             lw=2, ms=6, capsize=3, label="cliff")
axA.errorbar(xc, [m for m, _ in mn], yerr=[s for _, s in mn], fmt="s-", color=GRAY,
             lw=2, ms=6, capsize=3, label="non-cliff")
for i, t in enumerate(taus):
    r = mc[i][0] / mn[i][0]
    axA.text(xc[i], mc[i][0] + 0.035, f"{r:.1f}×", ha="center", fontsize=8.5,
             color=RED, fontweight="bold")
axA.set_xticks(xc); axA.set_xticklabels([f"{t:.2f}" for t in taus])
axA.set_xlabel("potency threshold τ (dataset y-quantile)")
axA.set_ylabel("under-estimation rate among true actives")
axA.set_title("Global conformal (nominal 90%)")
axA.legend(frameon=False, loc="upper left", fontsize=8)
axA.set_ylim(0, 0.62)
letter(axA, "A")

# ---- Panel B: method comparison at tau=0.75 ----
t0 = 0.75
order = ["mondrian_wstd", "global", "mondrian_rfstd", "mondrian_nb"]
labels = {"mondrian_wstd": "← d_wstd\n(safe)", "global": "global\n(no cond.)",
          "mondrian_rfstd": "← rf_std\n(model var.)", "mondrian_nb": "← d_nb\n(density)"}
cols = {"mondrian_wstd": GREEN, "global": GRAY, "mondrian_rfstd": RED, "mondrian_nb": RED}
xc2 = np.arange(len(order))
vals, errs, cs = [], [], []
for m in order:
    mm, ss = ms(R[(R.method == m) & (R.tau_q == t0) & (R.group == "cliff")], "under_rate")
    vals.append(mm); errs.append(ss); cs.append(cols[m])
axB.bar(xc2, vals, color=cs, width=0.62, edgecolor="white", linewidth=0.7, zorder=3)
axB.errorbar(xc2, vals, yerr=errs, fmt="none", ecolor="#333333", elinewidth=0.9,
             capsize=3, zorder=4)
nn_ref = ms(R[(R.method == "global") & (R.tau_q == t0) & (R.group == "noncliff")], "under_rate")[0]
axB.axhline(nn_ref, color="#333333", ls="--", lw=1.0, zorder=2)
axB.text(len(order) - 0.5, nn_ref + 0.008, "non-cliff baseline", ha="right", fontsize=7.5,
         color="#333333")
axB.set_xticks(xc2); axB.set_xticklabels([labels[m] for m in order], fontsize=7.8)
axB.set_ylabel("under-estimation rate among active cliffs")
axB.set_title("Same nominal coverage, τ = 0.75")
axB.set_ylim(0, 0.52)
letter(axB, "B")

fig.tight_layout()
out = "figures/fig11_decision_impact.png"
fig.savefig(out, bbox_inches="tight")
print("saved:", out)
