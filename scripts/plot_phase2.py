#!/usr/bin/env python
"""把 Phase 2 的交叉表画成图：为什么模型方差分区会砸掉悬崖组。

左图：每个分箱里 cliffB 的占比。
      如果信号能"看见"悬崖，占比应随分箱单调上升。d_wstd 是（1.0% -> 8.0%），
      rf_std 几乎不涨（2.9% -> 7.0%，但分布更平），d_nn_sim 甚至反向。
右图：每个分箱内 cliffB 的覆盖率，全局 CP vs 该分区的条件化 CP。
      低 rf_std 箱的分位数被大幅收紧（模型自认有把握的地方），而 cliffB 分子
      有相当一部分正落在那里 -> 覆盖率崩塌。

用法: python scripts/plot_phase2.py --dir results_phase2
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 图注用英文：一是论文图需要，二是 Linux 服务器上通常没有中文字体，
# 用中文会渲染成一堆方框（本地 Windows 有字体、服务器没有，很容易踩）。
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

C_GLOBAL = "#1f77b4"
C_RF = "#d62728"
C_WS = "#2ca02c"


def panel(ax, ct, cols, title, ylabel, xlabel):
    x = ct["bin"].to_numpy()
    for col, color, label, style in cols:
        ax.plot(x, ct[col].to_numpy(), style, color=color, label=label, markeredgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_phase2")
    ap.add_argument("--out", default="results_phase2/phase2_mechanism.png")
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    rf = pd.read_csv(d / "E_cross_rfstd.csv")
    ws = pd.read_csv(d / "F_cross_wstd.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    ax.plot(rf["bin"], rf["frac_cliffB"] * 100, "o-", color=C_RF,
            label="binned by rf_std", markeredgecolor="white")
    ax.plot(ws["bin"], ws["frac_cliffB"] * 100, "s-", color=C_WS,
            label="binned by d_wstd", markeredgecolor="white")
    base = (rf["n_cliffB"].sum() / rf["n"].sum()) * 100
    ax.axhline(base, ls="--", c="k", lw=1, label=f"overall cliff share {base:.1f}%")
    ax.set_title("(a) Does the signal rank activity cliffs?\ncliff share per bin")
    ax.set_xlabel("signal bin  (0 = smallest  ->  4 = largest)")
    ax.set_ylabel("cliff share in bin (%)")
    ax.set_xticks(rf["bin"])
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(rf["bin"], rf["cov_cliffB_global"], "o--", color=C_GLOBAL,
            label="global CP (bins by rf_std)", markeredgecolor="white")
    ax.plot(rf["bin"], rf["cov_cliffB_rfstd"], "o-", color=C_RF,
            label="conditioned on rf_std", markeredgecolor="white")
    ax.plot(ws["bin"], ws["cov_cliffB_wstd"], "s-", color=C_WS,
            label="conditioned on d_wstd", markeredgecolor="white")
    ax.axhline(0.9, ls=":", c="k", lw=1.2, label="nominal 0.90")
    ax.set_title("(b) Which partition breaks the cliff group?\ncliff coverage per bin")
    ax.set_xlabel("signal bin  (0 = smallest  ->  4 = largest)")
    ax.set_ylabel("cliff coverage in bin")
    ax.set_xticks(rf["bin"])
    ax.set_ylim(0.2, 1.0)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Cliff-group coverage: 0.723 (global)  ->  0.669 (conditioned on rf_std)  "
        "vs  0.742 (conditioned on d_wstd)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
