#!/usr/bin/env python
"""Fig 8：阈值敏感性热图 —— 核心结论是否只对默认阈值成立。

左图：各分区变量在各配置下对【悬崖组覆盖率】的影响（delta_mean）。
      颜色即符号与量级；斜线阴影 = 该配置不可用（数据集数不足）。
右图：方向一致性（多少个数据集上为正 / 有多少个数据集）。

用法: python scripts/plot_sensitivity.py --csv results_sens/SENSITIVITY.csv --out results_sens/fig8.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

# 配置列序（base 放最左，sim095 标记为不可用）
CFG_ORDER = ["base", "sim075", "sim095", "act050", "act150", "bins03", "bins08"]
CFG_LABEL = {
    "base": "base\nsim.85/act1.0/bins5",
    "sim075": "sim=0.75",
    "sim095": "sim=0.95\n(1 dataset)",
    "act050": "act=0.5",
    "act150": "act=1.5\n(9 datasets)",
    "bins03": "n_bins=3",
    "bins08": "n_bins=8",
}
VAR_ORDER = ["mondrian_nnsim", "mondrian_wstd", "mondrian_std", "mondrian_iqr",
             "mondrian_pair", "mondrian_pairgap", "mondrian_nb", "mondrian_rfstd"]
VAR_LABEL = {
    "mondrian_nnsim": "NN similarity",
    "mondrian_wstd": "label dispersion (wstd)",
    "mondrian_std": "label std",
    "mondrian_iqr": "label IQR",
    "mondrian_pair": "cliff pair present",
    "mondrian_pairgap": "max label gap in nbhd",
    "mondrian_nb": "# training neighbours",
    "mondrian_rfstd": "model-side variance",
}
UNUSABLE = {"sim095"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="fig8.png")
    args = ap.parse_args()

    A = pd.read_csv(args.csv)
    A = A[A["method"].isin(VAR_ORDER) & A["config"].isin(CFG_ORDER)]

    piv = A.pivot_table(index="method", columns="config", values="delta_mean")
    piv = piv.reindex(index=VAR_ORDER, columns=CFG_ORDER)
    con = A.pivot_table(index="method", columns="config", values="n_ds_delta_pos")
    nds = A.pivot_table(index="method", columns="config", values="n_ds")
    con = con.reindex(index=VAR_ORDER, columns=CFG_ORDER)
    nds = nds.reindex(index=VAR_ORDER, columns=CFG_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---------------- 左：delta 热图 ----------------
    ax = axes[0]
    M = piv.to_numpy(float)
    vmax = np.nanmax(np.abs(M))
    # 标准语义：红 = 有害（负），蓝 = 改善（正）。用 RdBu 而非 RdBu_r。
    im = ax.imshow(M * 100, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0, vmin=-vmax*100, vmax=vmax*100),
                   aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            cfg = CFG_ORDER[j]
            if cfg in UNUSABLE:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5, color="#666666")
                continue
            ax.text(j, i, f"{v*100:+.1f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if abs(v) > 0.03 else "black")
    # 不可用配置整列打斜线
    for j, cfg in enumerate(CFG_ORDER):
        if cfg in UNUSABLE:
            ax.axvspan(j - 0.5, j + 0.5, color="white", alpha=0.72, hatch="///", zorder=2)
    ax.set_xticks(range(len(CFG_ORDER)))
    ax.set_xticklabels([CFG_LABEL[c] for c in CFG_ORDER], fontsize=7.5)
    ax.set_yticks(range(len(VAR_ORDER)))
    ax.set_yticklabels([VAR_LABEL[v] for v in VAR_ORDER], fontsize=8.5)
    ax.set_title("A  Effect on cliff-group coverage (pp)\n"
                 "blue = improves the cliff group, red = harms it", fontsize=9.5)
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="Δ cliff coverage (pp)")

    # ---------------- 右：方向一致性 ----------------
    ax = axes[1]
    C = (con / nds.replace(0, np.nan)).to_numpy(float)
    Cm = np.ma.masked_invalid(C)
    im2 = ax.imshow(Cm, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            cfg = CFG_ORDER[j]
            if cfg in UNUSABLE or not np.isfinite(C[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5, color="#666666")
                continue
            n_pos, n_tot = int(con.to_numpy()[i, j]), int(nds.to_numpy()[i, j])
            ax.text(j, i, f"{n_pos}/{n_tot}", ha="center", va="center", fontsize=7.5,
                    color="black")
    for j, cfg in enumerate(CFG_ORDER):
        if cfg in UNUSABLE:
            ax.axvspan(j - 0.5, j + 0.5, color="white", alpha=0.72, hatch="///", zorder=2)
    ax.set_xticks(range(len(CFG_ORDER)))
    ax.set_xticklabels([CFG_LABEL[c] for c in CFG_ORDER], fontsize=7.5)
    ax.set_yticks(range(len(VAR_ORDER)))
    ax.set_yticklabels([VAR_LABEL[v] for v in VAR_ORDER], fontsize=8.5)
    ax.set_title("B  Direction consistency\n"
                 "(# datasets where the effect is positive / total)", fontsize=9.5)
    fig.colorbar(im2, ax=ax, shrink=0.85, pad=0.02, label="consistency")

    fig.suptitle(
        "Do the conclusions survive threshold changes?  "
        "15 ChEMBL assays x 5 splits per config, nominal 90% coverage",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(args.out, dpi=190)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
