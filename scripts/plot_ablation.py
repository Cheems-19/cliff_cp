#!/usr/bin/env python
"""分区变量消融图：收益（悬崖组覆盖率变化）vs 代价（平均区间宽度变化）。

一张图回答"该用哪个分区变量"：
  左上角（收益正、代价小）是最优区；右下角（代价大、收益负）是最差。
  同时标注每个分区变量的"看见悬崖 AUC"，看它能否解释位置。

用法: python scripts/plot_ablation.py --csv results_ablation/PHASE1_ABLATION.csv \
        --out results_ablation/ablation.png
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 分区变量家族：模型侧 / 密度与相似度侧 / 邻域矛盾侧 / 标签离散度侧
FAMILY = {
    "mondrian_rfstd": ("model-side (rf variance)", "#d62728"),
    "mondrian_nb": ("density-side (n neighbours)", "#ff7f0e"),
    "mondrian_nnsim": ("similarity-side (NN sim)", "#ff7f0e"),
    "mondrian_pair": ("neighbourhood-conflict", "#9467bd"),
    "mondrian_pairgap": ("neighbourhood-conflict", "#9467bd"),
    "mondrian_iqr": ("label-dispersion family", "#2ca02c"),
    "mondrian_std": ("label-dispersion family", "#2ca02c"),
    "mondrian_wstd": ("label-dispersion family", "#2ca02c"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_ablation/PHASE1_ABLATION.csv")
    ap.add_argument("--out", default="results_ablation/ablation.png")
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    d["family"] = d["method"].map(lambda m: FAMILY.get(m, ("other", "#888"))[0])
    d["color"] = d["method"].map(lambda m: FAMILY.get(m, ("other", "#888"))[1])
    d["label"] = d["method"].str.replace("mondrian_", "", regex=False)

    fig, ax = plt.subplots(figsize=(8.6, 6.0))

    for fam, sub in d.groupby("family"):
        ax.scatter(sub["width_delta"], sub["delta_mean"], s=110,
                   c=sub["color"].iloc[0], label=fam, zorder=3,
                   edgecolors="white", linewidths=1.2)

    for _, r in d.iterrows():
        ax.annotate(
            r["label"], (r["width_delta"], r["delta_mean"]),
            textcoords="offset points", xytext=(9, 6), fontsize=10,
        )

    ax.axhline(0, ls="--", c="k", lw=1)
    ax.axvline(0, ls="--", c="k", lw=1)
    # 必须显式设定坐标范围：否则 rf_std 这类"区间更窄但悬崖组更惨"的点会被自动边界裁掉，
    # 而它恰恰是图里最关键的对照。
    ax.set_xlim(d["width_delta"].min() - 0.012, d["width_delta"].max() + 0.012)
    ax.set_ylim(d["delta_mean"].min() - 0.008, d["delta_mean"].max() + 0.005)
    ax.set_xlabel("average interval-width change  (cost; negative = cheaper)")
    ax.set_ylabel("activity-cliff group coverage change\n(benefit; positive = safer)")
    ax.set_title(
        "Which partition variable should you use?\n"
        "top-left = better coverage AND cheaper intervals", fontsize=11,
    )
    # 图例放左上：左下会正好盖住 rf_std（代价最小、悬崖组最差）那个点
    ax.legend(fontsize=9, loc="upper left", framealpha=0.92)
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
