#!/usr/bin/env python
"""跨表示对比图：同一个分区变量，在两种完全不同的分子表示下是否给出同样结论。

左：8 个分区变量的悬崖组覆盖率变化，ECFP4 vs RDKit 2D 描述符，成对柱状。
右：错配量（预测失效能力 − 看见悬崖能力），两种表示对比。

判读：如果两套表示的柱高与符号一致，说明结论不是指纹表示的产物。
用法: python scripts/plot_crossarch.py --ecfp results_ablation/PHASE1_ABLATION.csv \
      --alt results_xa_rdkit/PHASE1_ABLATION.csv --out results_xa_rdkit/crossarch.png
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

C_ECFP = "#1f77b4"
C_ALT = "#d62728"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecfp", default="results_ablation/PHASE1_ABLATION.csv")
    ap.add_argument("--alt", default="results_xa_rdkit/PHASE1_ABLATION.csv")
    ap.add_argument("--alt-label", default="RDKit 2D descriptors")
    ap.add_argument("--out", default="results_xa_rdkit/crossarch.png")
    args = ap.parse_args()

    a = pd.read_csv(args.ecfp)[["method", "delta_mean"]].rename(columns={"delta_mean": "ecfp"})
    b = pd.read_csv(args.alt)[["method", "delta_mean"]].rename(columns={"delta_mean": "alt"})
    d = a.merge(b, on="method", how="inner")
    d["short"] = d["method"].str.replace("mondrian_", "", regex=False)
    d = d.sort_values("ecfp", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [3, 1.35]})

    ax = axes[0]
    x = np.arange(len(d))
    w = 0.38
    ax.bar(x - w / 2, d["ecfp"], w, label="ECFP4 (2048 bits)", color=C_ECFP, alpha=0.9)
    ax.bar(x + w / 2, d["alt"], w, label=args.alt_label, color=C_ALT, alpha=0.9)
    ax.axhline(0, ls="--", c="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(d["short"], rotation=25, ha="right")
    ax.set_ylabel("activity-cliff group coverage change")
    ax.set_title("(a) Same partition variable, two molecular representations")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    for i, (e, s) in enumerate(zip(d["ecfp"], d["alt"])):
        if np.sign(e) != np.sign(s):
            ax.annotate("sign flip", (i, min(e, s) - 0.004), ha="center",
                        fontsize=8, color="#b30000")

    ax = axes[1]
    # 错配量对比
    try:
        m1 = pd.read_csv("results_phase2/C_misalignment.csv")
        m2 = pd.read_csv("results_xa_rdkit/C_misalignment.csv")
    except Exception:
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"[saved] {args.out} (misalignment panel skipped)")
        return
    m = m1[["signal", "gap_predict_minus_see"]].merge(
        m2[["signal", "gap_predict_minus_see"]], on="signal", suffixes=("_ecfp", "_alt")
    ).sort_values("gap_predict_minus_see_ecfp", ascending=False)
    y = np.arange(len(m))
    ax.barh(y - w / 2, m["gap_predict_minus_see_ecfp"], w, color=C_ECFP, alpha=0.9,
            label="ECFP4")
    ax.barh(y + w / 2, m["gap_predict_minus_see_alt"], w, color=C_ALT, alpha=0.9,
            label=args.alt_label)
    ax.axvline(0, ls="--", c="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(m["signal"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("misalignment = AUC(predicts failure) - AUC(sees cliff)")
    ax.set_title("(b) Misalignment per signal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
