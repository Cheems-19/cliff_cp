#!/usr/bin/env python
"""Phase 6b 图：AD 定义换了阈值、换了代理，"域内悬崖更不可靠"是否还成立。

左图：三种失败率在 8 种 AD 界定下的对照
      域内悬崖 / 域内非悬崖 / 域外 —— 前两者的差距（≈20 pp）在 8 个配置里纹丝不动
右图：域内预警召回与相对随机的提升，同样在 8 个配置下稳定

用法: python scripts/plot_ad_sensitivity.py --dir results_ecfp_abl --out results_ecfp_abl/ad_sensitivity.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_CLIFF = "#d62728"
C_NON = "#1f77b4"
C_OUT = "#8c8c8c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="ad_sensitivity.png")
    args = ap.parse_args()

    A = pd.read_csv(os.path.join(args.dir, "AD_SENSITIVITY.csv"))
    A["short"] = A["proxy"].map(lambda s: "dist." if "距离" in s else "dens.")
    A["lab"] = A["short"] + "\n" + (A["ad_quantile"] * 100).round().astype(int).astype(str) + "%"

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))

    # ---------------- 左 ----------------
    ax = axes[0]
    x = np.arange(len(A))
    w = 0.27
    ax.bar(x - w, A["fail_in_ad_cliff"] * 100, w, color=C_CLIFF, edgecolor="white",
           label="in-AD $\\times$ activity cliff", zorder=3)
    ax.bar(x, (1 - A["cov_in_ad_noncliff"]) * 100, w, color=C_NON, edgecolor="white",
           label="in-AD $\\times$ non-cliff", zorder=3)
    ax.bar(x + w, A["fail_rate_out_ad"] * 100, w, color=C_OUT, edgecolor="white",
           label="rejected by the AD", zorder=3)
    for xi, (a, b) in enumerate(zip(A["fail_in_ad_cliff"], A["cov_in_ad_noncliff"])):
        ax.annotate(f"+{(a-(1-b))*100:.0f}pp", (xi - w, a * 100),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.5, color=C_CLIFF)
    ax.set_xticks(x)
    ax.set_xticklabels(A["lab"], fontsize=8)
    ax.set_xlabel("AD definition (distance / density type  ×  quantile cut)")
    ax.set_ylabel("failure rate (%)")
    ax.set_ylim(0, 36)
    ax.set_title("A  Inside the AD, cliffs fail ~4x more often\nthan non-cliffs — stable across all 8 cuts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    # ---------------- 右 ----------------
    ax = axes[1]
    ax.bar(x, A["in_ad_recall"] * 100, color=C_CLIFF, edgecolor="white", zorder=3,
           label="in-AD failure recall (dispersion flag, 20% budget)")
    ax.axhline(20, ls="--", c="k", lw=1.2)
    ax.text(0.02, 20.8, "random flagging (20%)", transform=ax.get_yaxis_transform(),
            fontsize=8)
    for xi, (r, l) in enumerate(zip(A["in_ad_recall"], A["in_ad_recall_lift"])):
        ax.text(xi, r * 100 + 0.6, f"{l:.2f}x", ha="center", fontsize=8, color=C_CLIFF)
    ax.set_xticks(x)
    ax.set_xticklabels(A["lab"], fontsize=8)
    ax.set_xlabel("AD definition (distance / density type  ×  quantile cut)")
    ax.set_ylabel("failure recall (%)")
    ax.set_ylim(0, 52)
    ax.set_title("B  The label-free dispersion flag keeps\nthe same edge (1.78–1.95x) inside the AD")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "Robustness of the applicability-domain result "
        "(70,170 molecules, 15 ChEMBL assays, 10 splits)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(args.out, dpi=190)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
