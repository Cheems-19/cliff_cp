#!/usr/bin/env python
"""Phase 6 图：适用域过滤器为什么抓不到活动悬崖失效，以及域内该怎么预警。

三个面板：
  A 覆盖率沿"最近邻相似度"分层：越稠密覆盖率越高（AD 的默认结论），
    但活动悬崖只存在于最稠密的两层 —— AD 的"稠密=可靠"在这里失效
  B 覆盖率 2×2：AD 内 / 外 × 悬崖 / 非悬崖。域外一个悬崖都没有，
    而域内悬崖的覆盖率比域内非悬崖低 22 个百分点
  C 同等预警预算下的失效召回：域内用离散度预警相对随机的提升

用法:
    python scripts/plot_ad.py --dir results_ecfp_abl --out results_ecfp_abl/ad_flagging.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_SPARSE = "#8c8c8c"
C_DENSE = "#1f77b4"
C_CLIFF = "#d62728"
C_AD = "#7f7f7f"
C_DISP = "#2ca02c"
C_BOTH = "#7f4fbf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="ad_flagging.png")
    ap.add_argument("--flag-frac", type=float, default=0.20)
    args = ap.parse_args()

    S = pd.read_csv(os.path.join(args.dir, "AD_BY_SIMBIN.csv"))
    T = pd.read_csv(os.path.join(args.dir, "AD_2x2.csv"))
    C = pd.read_csv(os.path.join(args.dir, "AD_RECALL_CURVES.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6))

    # ---------------- A ----------------
    ax = axes[0]
    x = S["sim_bin"].to_numpy(int)
    cov = S["coverage"].to_numpy(float) * 100
    cl = S["cliffB_rate"].to_numpy(float)
    ax.bar(x, cov, color=[C_SPARSE, C_SPARSE, C_DENSE, C_DENSE, C_DENSE],
           edgecolor="white", zorder=3, label="coverage of all molecules")
    for xi, ci in zip(x, cl):
        if ci > 0:
            ax.annotate(f"cliffs\n{ci:.1f}%", (xi, cov[xi]), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=8.5, color=C_CLIFF)
    ax.axhline(90, ls="--", c="k", lw=1.1)
    ax.text(0.02, 90.6, "nominal 90%", fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(["0\n(sparsest)", "1", "2", "3", "4\n(densest)"], fontsize=8.5)
    ax.set_xlabel("nearest-neighbour similarity quintile")
    ax.set_ylabel("coverage / rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title("A  Denser = better covered,\nbut activity cliffs live only in bins 3–4")
    ax.grid(alpha=0.25, axis="y")

    # ---------------- B ----------------
    ax = axes[1]
    gname = T.columns[0]
    lab, val, col = [], [], []
    for g, cv in zip(T[gname].astype(str), T["coverage"].astype(float)):
        pretty = (g.replace("AD 内", "in-AD").replace("AD 外", "out-AD")
                   .replace("悬崖(cliffB)", "cliff").replace("非悬崖", "non-cliff"))
        lab.append(pretty)
        val.append(cv * 100)
        col.append(C_CLIFF if ("cliff" in pretty and "non-cliff" not in pretty) else C_AD)
    xs = np.arange(len(val))
    ax.bar(xs, val, color=col, edgecolor="white", zorder=3)
    for xi, vi in zip(xs, val):
        ax.text(xi, vi + 1.0, f"{vi:.1f}", ha="center", fontsize=9)
    ax.axhline(90, ls="--", c="k", lw=1.1)
    ax.set_xticks(xs)
    ax.set_xticklabels([l.replace(" × ", "\n× ") for l in lab], fontsize=8.5)
    ax.set_ylabel("coverage (%)")
    ax.set_ylim(0, 108)
    ax.set_title("B  All cliffs are inside the AD —\nand cost 22 pp of coverage there")
    ax.annotate("no cliff exists\noutside the AD", xy=(0.5, 0.06),
                xycoords="axes fraction", ha="center", fontsize=8.5,
                color=C_CLIFF,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_CLIFF, lw=0.8))
    ax.grid(alpha=0.25, axis="y")

    # ---------------- C ----------------
    ax = axes[2]
    f = args.flag_frac
    C["s"] = C["rule"].astype(str).str.replace("（.*）", "", regex=True).str.strip()
    sub = C[np.isclose(C["flag_frac"], f)]
    # 图内一律用英文：matplotlib 在缺中文字体时会把标签渲染成方框，
    # 而且论文图本来也该是 ASCII（见 skill 里的对应条目）。
    EN = {"AD 规则": "AD rule\n(global)",
          "离散度规则": "dispersion\n(global)",
          "合并": "combined\n(global)"}
    names, vals, cols = [], [], []
    for s, v in zip(sub["s"], sub["failure_recall"]):
        names.append(EN.get(s, s))
        vals.append(v * 100)
        cols.append(C_DISP if "离散度" in s else (C_BOTH if "合并" in s else C_AD))
    # 域内离散度（来自 ad_flagging.log 第 ④ 节实测：0.3781）
    # 域内离散度（来自 ad_flagging.log 第 ④ 节实测：0.3683）
    names.append("dispersion\n(in-AD only)")
    vals.append(36.83)
    cols.append(C_DISP)
    xs = np.arange(len(vals))
    ax.bar(xs, vals, color=cols, edgecolor="white", zorder=3)
    ax.axhline(f * 100, ls="--", c="k", lw=1.2)
    ax.text(0.02, f * 100 + 1.5, f"random flagging ({f:.0%})",
            transform=ax.get_yaxis_transform(), fontsize=8)
    for xi, vi in zip(xs, vals):
        ax.text(xi, vi + 0.8, f"{vi:.1f}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("failure recall (%)")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title(f"C  With a {f:.0%} review budget:\ndispersion helps where the AD is blind")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "Applicability-domain filtering is blind to activity-cliff failures "
        "— 70,170 molecules, 15 ChEMBL assays, 10 splits",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.out, dpi=190)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
