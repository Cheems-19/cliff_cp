# -*- coding: utf-8 -*-
"""Fig 12 — 覆盖转移恒等式的可部署诊断验证。
x = 净收紧质量 net_mag（无标签，仅用校准集 + 信号 + 悬崖掩码）
y = 观测到的悬崖组覆盖变化 ΔCov（pp）
Spearman ρ = 0.865 (n=16)。300 dpi。
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

H = pd.read_csv("results_phase14/TRANSFER_HARM.csv")
a = H.groupby(["run", "signal"]).mean(numeric_only=True).reset_index()
a["y"] = 100 * a.d_cov_cliff

COL = {"rf_ecfp": "#2c5f8a", "rf_rdkit": "#e08a1e"}
MK = {"d_wstd": "o", "d_std": "s", "d_iqr": "^", "d_nb": "D", "d_nn_sim": "v",
      "d_pair_gap": "P", "d_pair": "X", "rf_std": "*"}

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for run, g in a.groupby("run"):
    for _, r in g.iterrows():
        ax.scatter(r.net_mag, r.y, s=(150 if r.signal == "rf_std" else 52),
                   marker=MK[r.signal], color=COL[run], alpha=0.9, zorder=3,
                   edgecolor="white", linewidth=0.6)
# 标注（对每个信号取两个表示的中点）
OFFS = {"d_wstd": (7, 5), "d_std": (7, -15), "d_iqr": (7, 5), "d_nb": (7, 5),
        "d_nn_sim": (7, -15), "d_pair_gap": (7, 5), "d_pair": (7, -15),
        "rf_std": (7, 5)}
for sig, g in a.groupby("signal"):
    ax.annotate(sig, (g.net_mag.mean(), g.y.mean()), textcoords="offset points",
                xytext=OFFS.get(sig, (7, 5)), fontsize=8, color="#333333")

ax.axhline(0, color="#555555", lw=0.9, ls="--", zorder=1)
ax.axvline(0, color="#555555", lw=0.9, ls="--", zorder=1)
rho = a.net_mag.corr(a.y, method="spearman")
ax.text(0.02, 0.03, f"Spearman ρ = {rho:.3f}  (n={len(a)})", transform=ax.transAxes,
        fontsize=9.5, color="#1a1a1a", fontweight="bold")
ax.text(0.02, 0.10, "net tightening  →  harm", transform=ax.transAxes,
        fontsize=8, color="#8a2b2b")
ax.text(0.70, 0.985, "net widening  →  benefit", transform=ax.transAxes,
        fontsize=8, color="#1e6b3a", va="top")
ax.set_xlabel("net relative tightening of the cliff group, $\\mathrm{net\\_mag}$  (label-free)")
ax.set_ylabel("$\\Delta$ cliff coverage vs global (pp)")
ax.set_title("A label-free predictor of coverage transfer (identity-validated)")
handles = [plt.Line2D([], [], marker="o", ls="", color=COL["rf_ecfp"], label="ECFP4 predictor"),
           plt.Line2D([], [], marker="o", ls="", color=COL["rf_rdkit"], label="RDKit-2D predictor")]
ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=8)
fig.tight_layout()
out = "figures/fig12_transfer_diagnostic.png"
fig.savefig(out, bbox_inches="tight")
print("saved:", out, "| spearman:", round(float(rho), 3))
