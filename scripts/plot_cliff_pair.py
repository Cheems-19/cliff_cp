# -*- coding: utf-8 -*-
"""图 14：悬崖对实例 —— 两个 ECFP4 指纹完全相同（sim=1.0）的分子，
效力相差 2.63 个对数单位；模型给出同一预测与同一 90% 区间，
高效力分子区间失效、低效力邻居被覆盖。

素材：results_server/phase18_x/phase18_cliffpair/（服务器 RDKit 绘制的结构 PNG +
pair_info.csv）。本脚本只做排版组合。
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
SRC = "results_server/phase18_x/phase18_cliffpair"

info = pd.read_csv(os.path.join(SRC, "pair_info.csv")).iloc[0]
RED = "#C0392B"
BLUE = "#2471A3"
GRAY = "#7F8C8D"

fig = plt.figure(figsize=(8.6, 3.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.25)

# --- A / B：结构 ---
axA = fig.add_subplot(gs[0])
axA.imshow(mpimg.imread(os.path.join(SRC, "molA_highCliff.png")))
axA.set_title(f"Molecule A (test, cliff)\ntrue pAct = {info.y_i:.2f}", fontsize=10)
axA.axis("off")

axB = fig.add_subplot(gs[1])
axB.imshow(mpimg.imread(os.path.join(SRC, "molB_trainNeighbour.png")))
axB.set_title(f"Molecule B (train neighbour)\ntrue pAct = {info.y_j:.2f}", fontsize=10)
axB.axis("off")

# --- C：区间对比 ---
axC = fig.add_subplot(gs[2])
q = float(info.q_global)
pred = float(info.pred_i)
for x, y_true, covered, col in [(0, float(info.y_i), False, RED),
                                (1, float(info.y_j), True, BLUE)]:
    axC.plot([x, x], [pred - q, pred + q], color=GRAY, lw=10, alpha=0.35,
             solid_capstyle="butt", zorder=1)
    axC.scatter([x], [y_true], s=90, color=col, zorder=3, edgecolor="black", linewidth=0.6)
    axC.annotate(f"{y_true:.2f}", (x, y_true), xytext=(12, -4),
                 textcoords="offset points", fontsize=9, color=col, fontweight="bold")
axC.axhline(pred, color=GRAY, lw=1, ls="--")
axC.annotate(f"prediction {pred:.2f}", (1.02, pred), fontsize=8.5, color=GRAY,
             ha="left", va="center")
axC.set_xticks([0, 1])
axC.set_xticklabels(["A (cliff)", "B"], fontsize=10)
axC.set_xlim(-0.55, 1.75)
axC.set_ylim(4.6, 9.7)
axC.set_ylabel("pActivity", fontsize=10)
axC.set_title("Same 90% prediction interval\n(identical ECFP4, Tanimoto = 1.00)",
              fontsize=10)
axC.annotate("interval miss", (0, float(info.y_i)), xytext=(-58, 16),
             textcoords="offset points", fontsize=8.5, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1))

fig.subplots_adjust(left=0.04, right=0.99, top=0.86, bottom=0.10, wspace=0.18)
out = "figures/fig14_cliff_pair.png"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"[saved] {out}")
