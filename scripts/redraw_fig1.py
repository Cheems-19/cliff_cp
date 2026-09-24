# -*- coding: utf-8 -*-
"""Fig 1 — 悬崖 vs 非悬崖覆盖率（单数据集示例）+ 合并森林图（出版质量，300 dpi）。
数据：results_server/COVERAGE.csv（逐数据集覆盖率）+ results_server/META.csv（随机效应合并）。
输出：figures/fig1_coverage_gap.png
"""
import os, sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

# 统一风格（与其余 fig2-10 一致）
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 600,
})

RED = "#c0392b"     # cliff / harm
GRAY = "#7f8c8d"    # noncliff / neutral
BLUE = "#2c5f8a"    # secondary
GREEN = "#1e8449"   # aligned

def letter(ax, ch):
    ax.text(-0.13, 1.06, ch, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")

# ---------- 读取数据 ----------
cov_rows = []
with open("results_server/COVERAGE.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        r["coverage_mean"] = float(r["coverage_mean"])
        r["ci_lo"] = float(r["ci_lo"]); r["ci_hi"] = float(r["ci_hi"])
        cov_rows.append(r)

meta_rows = []
with open("results_server/META.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        for k in ("effect", "ci_lo", "ci_hi", "I2", "p_value", "n_datasets", "n_datasets_gap_pos"):
            r[k] = float(r[k])
        meta_rows.append(r)

group_defs = sorted({r["group_def"] for r in meta_rows})
methods = ["global", "cluster", "knn_weighted"]
GDEF_LABEL = {"defA": "Def A (dataset-level)",
              "defB": "Def B (mechanism)",
              "defC": "Def C (proxy)",
              "defB_oracle": "Def B (mechanism)",
              "defC_proxy": "Def C (proxy)"}
METH_LABEL = {"global": "global CP", "cluster": "cluster CP", "knn_weighted": "kNN-weighted CP"}

# ---------- Panel A：单数据集示例（CHEMBL233_Ki, defB） ----------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.0, 4.0),
                               gridspec_kw={"width_ratios": [1.0, 1.25]})
ds = "CHEMBL233_Ki"
x = np.arange(len(methods)); w = 0.36
for i, side in enumerate(["cliff", "noncliff"]):
    vals, err = [], []
    for m in methods:
        row = next(r for r in cov_rows
                   if r["dataset"] == ds and r["group_def"] == "defB_oracle"
                   and r["method"] == m and r["side"] == side)
        vals.append(row["coverage_mean"])
        err.append([row["coverage_mean"] - row["ci_lo"], row["ci_hi"] - row["coverage_mean"]])
    off = -w/2 if side == "cliff" else w/2
    color = RED if side == "cliff" else GRAY
    axA.bar(x + off, vals, width=w, color=color, edgecolor="white", linewidth=0.6,
            label="cliff" if side == "cliff" else "non-cliff", zorder=3)
    axA.errorbar(x + off, vals, yerr=np.array(err).T, fmt="none", ecolor="#333333",
                 elinewidth=0.9, capsize=2.5, zorder=4)
axA.axhline(0.90, color="#1a1a1a", linestyle="--", linewidth=0.9, zorder=2)
axA.text(len(methods)-0.42, 0.903, "nominal 0.90", fontsize=7.5, color="#1a1a1a", va="bottom")
axA.set_xticks(x); axA.set_xticklabels([METH_LABEL[m] for m in methods], fontsize=8)
axA.set_ylim(0.60, 1.0)
axA.set_ylabel("Coverage")
axA.set_title(f"{ds}, cliff def B")
axA.legend(frameon=False, loc="upper center", ncol=2, fontsize=8,
           columnspacing=1.2, handletextpad=0.4, borderaxespad=0.2)
letter(axA, "A")

# ---------- Panel B：合并森林图（随机效应） ----------
y, ylabels, colors = [], [], []
pos = 0
yticks, yticklabels = [], []
for gd in group_defs:
    for m in methods:
        row = next(r for r in meta_rows if r["group_def"] == gd and r["method"] == m)
        yticks.append(pos)
        yticklabels.append(f"{METH_LABEL[m]}")
        # 主行：defB + global 高亮
        colors.append(RED if (gd == "defB" and m == "global") else GRAY)
        row["_pos"] = pos
        pos += 1
    pos += 1  # 组间空行

for gd in group_defs:
    for m in methods:
        row = next(r for r in meta_rows if r["group_def"] == gd and r["method"] == m)
        p = row["_pos"]
        c = RED if (gd == "defB" and m == "global") else GRAY
        axB.errorbar(row["effect"], p,
                     xerr=[[row["effect"] - row["ci_lo"]], [row["ci_hi"] - row["effect"]]],
                     fmt="o", color=c, ecolor=c, elinewidth=1.2, capsize=3,
                     markersize=5, zorder=3)
axB.axvline(0, color="#1a1a1a", linewidth=0.8, linestyle="-", zorder=1)
axB.set_yticks(yticks)
axB.set_yticklabels(yticklabels, fontsize=8)
# 组标签放右侧
ymax = pos
for i, gd in enumerate(group_defs):
    yy = i * (len(methods) + 1) + 1
    axB.text(1.02, yy / ymax, GDEF_LABEL.get(gd, gd), transform=axB.get_yaxis_transform(),
             fontsize=7.5, va="center", ha="left", color="#333333")
axB.invert_yaxis()
axB.set_xlabel("Coverage gap, cliff vs non-cliff (pp)\nrandom-effects, 95% CI")
axB.set_xlim(-0.02, 0.28)
axB.xaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}")
axB.set_title("Merged across 8 datasets (8/8 direction-consistent)")
letter(axB, "B")

fig.tight_layout()
out = os.path.join("figures", "fig1_coverage_gap.png")
fig.savefig(out, bbox_inches="tight")
print("saved:", out)
print("defB global:", [ (r["effect"], r["ci_lo"], r["ci_hi"], r["I2"]) for r in meta_rows
                       if "defB" in r["group_def"] and r["method"]=="global"])
