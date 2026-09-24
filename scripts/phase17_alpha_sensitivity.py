# -*- coding: utf-8 -*-
"""C1：α 敏感性 —— 用服务器现成的 alpha 扫描结果（results/alpha_007..020）计算
悬崖覆盖差距随 α 的变化。数据源：results_server/alpha_x/alpha_*/METHODS.csv
（phase1_signal 产物：seed, method, group_def, side, coverage, dataset）。

结论（cliffB 机制口径，macro over 15 数据集，4 种子）：
  悬崖差距随 α 单调增长（global +13.2 → +30.1 pp；cliffA 口径 +7.3 → +17.9），
  α=0.07 即 14/15 方向一致；边际覆盖始终贴名义。
  → 现象不是 α=0.10 单点，且越保守的区间差距越大（悬崖失效集中于重尾）。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

C = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(C, "results_server", "alpha_x")
OUT = os.path.join(C, "results_server", "C1_ALPHA_SENSITIVITY.txt")
ALPHAS = ["007", "010", "013", "016", "020"]
METHODS = ["global", "mondrian_wstd", "mondrian_nb", "mondrian_rfstd"]


def main() -> None:
    lines = []

    def w(s=""):
        lines.append(str(s))
        print(s)

    frames = []
    for a in ALPHAS:
        d = pd.read_csv(os.path.join(SRC, f"alpha_{a}", "METHODS.csv"))
        d["alpha"] = int(a) / 100
        frames.append(d)
    A = pd.concat(frames, ignore_index=True)
    A = A[A.group_def.isin(["cliffB_oracle", "cliffA"])]
    piv = A.pivot_table(index=["alpha", "dataset", "seed", "method", "group_def"],
                        columns="side", values="coverage").reset_index()
    piv["gap"] = (piv["noncliff"] - piv["cliff"]) * 100
    piv = piv.dropna(subset=["gap"])

    w("=== C1 悬崖覆盖差距（macro pp）随 α ===")
    for gd, tag in [("cliffB_oracle", "cliffB（机制口径）"), ("cliffA", "cliffA（划分无关）")]:
        w(f"--- {tag} ---")
        w(f"{'α':>5s}" + "".join(f"{m:>18s}" for m in METHODS))
        for a in [0.07, 0.10, 0.13, 0.16, 0.20]:
            line = f"{a:>5.2f}"
            for m in METHODS:
                s = piv[(piv.alpha == a) & (piv.method == m) & (piv.group_def == gd)]
                per = s.groupby("dataset").gap.mean()
                pos = int((per > 0).sum())
                try:
                    wl = stats.wilcoxon(per, zero_method="wilcox").pvalue
                except Exception:
                    wl = float("nan")
                line += f"  {per.mean():+7.1f} ({pos:>2d}/{len(per):<2d}, p={wl:.0e})"
            w(line)
    w("")
    w("=== 边际覆盖（global, side=all）按 α ===")
    for a in [0.07, 0.10, 0.13, 0.16, 0.20]:
        s = A[(A.alpha == a) & (A.method == "global") & (A.side == "all")
              & (A.group_def == "cliffB_oracle")]
        w(f"   α={a:.2f}  cov_all = {s.coverage.mean():.4f}  (名义 {1 - a:.2f})")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
