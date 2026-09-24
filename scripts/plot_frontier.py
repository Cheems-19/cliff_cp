#!/usr/bin/env python
"""画"代价—收益"前沿图：校准分区的收益有多少是买来的宽度？

左图：原始视图。x = 平均区间宽度相对 global 的比值，y = 悬崖组覆盖率相对 global 的变化。
      上升的 γ 曲线说明"收益随宽度单调上升" —— 无法区分是切分真的更好，还是只是更宽。
右图：匹配宽度视图。把每条规则的 q 整体缩放至与 global 同宽，再看悬崖组差异。
      所有规则都塌到 0 附近或以下 —— 不存在免费的效率增益。

用法: python scripts/plot_frontier.py --csv results_ecfp_abl/FRONTIER_GAMMA.csv \
        --paired results_ecfp_abl/FRONTIER_PAIRED.csv --out results_ecfp_abl/frontier.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 统一配色（论文图用英文标注）
C_REF = "#333333"
C_FLOOR = "#d62728"
C_WS = "#2ca02c"
C_HARM = "#7f4fbf"
C_NS = "#8c8c8c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="FRONTIER_GAMMA.csv")
    ap.add_argument("--paired", required=True, help="FRONTIER_PAIRED.csv")
    ap.add_argument("--out", default="frontier.png")
    args = ap.parse_args()

    G = pd.read_csv(args.csv)
    P = pd.read_csv(args.paired)
    P = P.set_index("rule")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))

    # ---------------- 左：原始视图 ----------------
    ax = axes[0]
    x = G["w_ratio"].to_numpy(float) * 100 - 100      # 宽度变化 %
    y = G["d_cliff_raw"].to_numpy(float) * 100        # 悬崖组变化 pp
    ax.plot(x, y, "o-", color=C_FLOOR, lw=2.0, ms=6,
            markeredgecolor="white", label="one-sided rule  max(q$_{bin}$, γ·q$_{global}$)")
    for gv, xi, yi in zip(G["gamma"], x, y):
        if gv in (0.0, 0.8, 1.0):
            ax.annotate(f"γ={gv:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(6, -12), fontsize=8.5, color=C_FLOOR)
    ax.axhline(0, ls="--", c=C_REF, lw=1.1)
    ax.axvline(0, ls="--", c=C_REF, lw=1.1)

    others = {
        "rfstd": ("model-side var. (misaligned)", C_HARM, "s"),
        "nb": ("# training neighbours", C_HARM, "^"),
        "wstd": ("label dispersion (aligned)", C_WS, "D"),
        "std": ("label std", C_WS, "v"),
        "iqr": ("label IQR", C_NS, "P"),
        "nnsim": ("NN similarity", C_NS, "X"),
    }
    for tag, (lab, col, mk) in others.items():
        if tag not in P.index:
            continue
        ax.scatter([P.loc[tag, "w_ratio"] * 100 - 100], [P.loc[tag, "d_cliff_raw"] * 100],
                   marker=mk, s=90, color=col, edgecolor="white", zorder=5,
                   label=lab if tag in ("rfstd", "wstd", "nb", "nnsim") else None)
    ax.set_xlabel("mean interval width vs global (%)")
    ax.set_ylabel("cliff-group coverage vs global (pp)")
    ax.set_title("① Raw view: any gain tracks interval width")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    # ---------------- 右：匹配宽度视图 ----------------
    ax = axes[1]
    order = ["rfstd", "nb", "iqr", "wstd", "std", "nnsim", "floor_g1.00"]
    names = {
        "rfstd": "model-side var.", "nb": "# neighbours", "iqr": "label IQR",
        "wstd": "label dispersion", "std": "label std", "nnsim": "NN similarity",
        "floor_g1.00": "one-sided rule (γ=1)",
    }
    vals, errs, cols, labs = [], [], [], []
    for tag in order:
        if tag not in P.index:
            continue
        v = P.loc[tag, "d_cliff_mtd"] * 100
        vals.append(v)
        errs.append(0.0)
        cols.append(C_FLOOR if tag == "floor_g1.00" else
                    (C_HARM if tag in ("rfstd", "nb") else
                     (C_WS if tag in ("wstd", "std") else C_NS)))
        labs.append(names[tag])
    xs = np.arange(len(vals))
    bars = ax.bar(xs, vals, color=cols, edgecolor="white", zorder=3)
    ax.axhline(0, c=C_REF, lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, rotation=22, ha="right", fontsize=8.5)
    ax.set_ylabel("cliff-group coverage vs global (pp)")
    ax.set_title("② Matched width: no free lunch — all gains vanish,\nharm survives")
    # 标注显著性 + 方向一致性
    tags_used = [t for t in order if t in P.index]
    for i, tag in enumerate(tags_used):
        p = P.loc[tag, "p_mtd"]
        star = ""
        if np.isfinite(p):
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        pos = P.loc[tag, "pos_mtd"]
        txt = f"{star}\n{pos}"
        if vals[i] >= 0:
            ax.text(i, vals[i] + 0.12, txt, ha="center", va="bottom", fontsize=7.5)
        else:
            ax.text(i, vals[i] - 0.12, txt, ha="center", va="top", fontsize=7.5)
    span = max(abs(min(vals)), abs(max(vals)))
    ax.set_ylim(-span * 1.35, span * 0.85)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "Is partitioning the calibration set worth it?  "
        "15 ChEMBL assay datasets x 10 splits, nominal 90% coverage",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=190)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
