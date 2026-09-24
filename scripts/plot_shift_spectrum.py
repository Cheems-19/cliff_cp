#!/usr/bin/env python
"""Fig 13 —— 偏移强度谱（phase15）。

A) 悬崖覆盖差距 vs 划分协议（cliffA / cliffB，5 个协议按强度递增）
B) 边际覆盖 vs 划分协议（结构协议不塌、活性偏移崩塌；adaptive 部分救回）
C) 连续偏移强度：random 基准下按「到训练集最大相似度」十分位分层
   —— 覆盖率随相似度上升，但悬崖差距不缩小，且悬崖集中在最"可信"层。
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
})

HARM = "#b2182b"
BETTER = "#2166ac"
NEUTRAL = "#8c8c8c"
REF = "#333333"
ORANGE = "#ef8a62"
LIGHT = "#d9d9d9"

MS = pd.read_csv("results_server/phase15_METHODS.csv")
MID = pd.read_csv("results_server/phase15_actmid_x/phase15_actmid/METHODS.csv")
MS = pd.concat([MS, MID], ignore_index=True)
MA = pd.read_csv("results_server/phase15_extract/phase15_adaptive/METHODS.csv")
MA = pd.concat([MA, MID[MID.method == "adaptive"]], ignore_index=True)
PROTOS = ["random", "scaffold", "scaffold_gen", "act_mid20", "act_shift40", "act_shift20"]
LABELS = ["random", "scaffold (exact)", "scaffold (generic)",
          "act mid 20%", "act shift 40%", "act shift 20%"]


def macro_gap(ms, proto, cc, cn):
    s = ms[(ms.method == "global") & (ms.protocol == proto)]
    per = s.groupby("dataset")[[cc, cn]].mean()
    return (per[cn] - per[cc]).mean() * 100


def macro_cov(ms, proto, method="global"):
    s = ms[(ms.method == method) & (ms.protocol == proto)]
    return s.cov_all.mean()


fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), gridspec_kw={"width_ratios": [1.15, 1.05, 1.25]})

# ---------------- A) 悬崖差距 vs 协议 ----------------
ax = axes[0]
x = np.arange(len(PROTOS))
ga = [macro_gap(MS, p, "cov_cliffA", "cov_noncliffA") for p in PROTOS]
gb = [macro_gap(MS, p, "cov_cliffB", "cov_noncliffB") for p in PROTOS]
w = 0.38
ax.bar(x - w / 2, ga, w, color=NEUTRAL, edgecolor="white", zorder=3, label="cliff A (dataset-level)")
ax.bar(x + w / 2, gb, w, color=HARM, edgecolor="white", zorder=3, label="cliff B (mechanism)")
for xi, v in zip(x, ga):
    ax.text(xi - w / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=7.5, color=NEUTRAL)
for xi, v in zip(x, gb):
    ax.text(xi + w / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=7.5, color=HARM)
ax.axvspan(3.5, 5.5, color="#f2f2f2", zorder=0)
ax.text(4.5, 22.6, "activity shift", ha="center", fontsize=8, color=REF)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=7, rotation=30, ha="right")
ax.set_ylabel("cliff coverage gap (pp)")
ax.set_ylim(0, 31)
ax.set_title("A  Gap shrinks with structural shift,\npersists under activity shift", fontsize=9)
ax.legend(loc="upper left", framealpha=0.9, fontsize=7)
ax.grid(alpha=0.25, axis="y")

# ---------------- B) 边际覆盖 vs 协议 ----------------
ax = axes[1]
cov = [macro_cov(MS, p) for p in PROTOS]
cols = [NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, HARM, HARM]
bars = ax.bar(x, cov, 0.6, color=cols, edgecolor="white", zorder=3, label="global")
# adaptive（活性偏移下部分救回）
ad = [macro_cov(MA, p, "adaptive") if p.startswith("act_") else np.nan for p in PROTOS]
ax.bar(x, ad, 0.6, color=BETTER, alpha=0.85, edgecolor="white", zorder=4,
       label="adaptive (shift-aware)")
lab_pos = {4: (3.30, 0.46), 5: (4.35, 0.60)}  # 被叠加柱遮挡的 global 标签 → 引线放到空白区
for xi, v in zip(x, cov):
    if xi in lab_pos:
        ax.annotate(f"{v:.2f}", (xi - 0.18, v), xytext=lab_pos[xi], fontsize=7.5,
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#555555"))
    else:
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)
for xi, v in zip(x, ad):
    if np.isfinite(v):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5, color=BETTER)
ax.axhline(0.90, ls="--", c=REF, lw=1.1)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=7, rotation=30, ha="right")
ax.set_ylabel("marginal coverage")
ax.set_ylim(0, 1.32)
ax.set_title("B  Structural shift keeps the guarantee;\nactivity shift breaks it entirely", fontsize=9)
ax.legend(loc="upper right", framealpha=0.9, fontsize=7.5)
ax.grid(alpha=0.25, axis="y")

# ---------------- C) random 基准的相似度十分位曲线 ----------------
ax = axes[2]
import glob  # noqa: E402
fr = [pd.read_csv(f) for f in sorted(glob.glob(
    "results_server/phase15_extract/phase15_spectrum/dump/*_spectrum.csv.gz"))]
A = pd.concat(fr, ignore_index=True)
A = A[(A.protocol == "random") & np.isfinite(A.nn_sim_train)].copy()
A["decile"] = pd.qcut(A.nn_sim_train, 10, labels=False, duplicates="drop")
rows = []
for dq, sub in A.groupby("decile"):
    cl = sub[sub.cliffA]
    ncl = sub[~sub.cliffA]
    rows.append(dict(d=int(dq) + 1,
                     cov_all=sub.covered_global.mean(),
                     cov_cliff=cl.covered_global.mean() if len(cl) else np.nan,
                     cov_non=ncl.covered_global.mean() if len(ncl) else np.nan,
                     share=sub.cliffA.mean() * 100,
                     sim=sub.nn_sim_train.median()))
R = pd.DataFrame(rows).sort_values("d")
ax2 = ax.twinx()
ax2.bar(R.d, R.share, 0.62, color=LIGHT, edgecolor="white", zorder=1,
        label="cliff share (right)")
ax2.set_ylim(0, 62)
ax2.set_ylabel("cliff share in decile (%)", fontsize=8)
ax2.spines["right"].set_visible(True)
ax2.tick_params(labelsize=8)

ax.plot(R.d, R.cov_non, "-o", color=NEUTRAL, ms=4, lw=1.4, zorder=3, label="non-cliff")
ax.plot(R.d, R.cov_all, "-o", color=REF, ms=4, lw=1.4, zorder=4, label="all molecules")
ax.plot(R.d, R.cov_cliff, "-o", color=HARM, ms=4, lw=1.4, zorder=5, label="cliff (A)")
ax.axhline(0.90, ls="--", c=REF, lw=1.0, zorder=2)
ax.text(10.35, 0.888, "nominal 90%", fontsize=7, ha="right", va="top", color=REF)
ax.set_xticks(R.d)
ax.set_xlabel("decile of max Tanimoto similarity to training set")
ax.set_ylabel("coverage")
ax.set_ylim(0.45, 1.02)
ax.set_title("C  Coverage rises with similarity,\nthe cliff gap does not close", fontsize=9)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.34),
          ncol=2, frameon=False, fontsize=7)
ax.grid(alpha=0.25, axis="y")

for ax_, ch in zip(axes, "ABC"):
    ax_.text(-0.16, 1.04, ch, transform=ax_.transAxes, fontsize=12, fontweight="bold")

fig.suptitle("Shift-intensity spectrum: the cliff gap is orthogonal to the distance axis "
             "(15 ChEMBL datasets, 3 seeds, RF, alpha = 0.10)", fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("figures/fig13_shift_spectrum.png")
plt.close(fig)
print("[saved] figures/fig13_shift_spectrum.png")
