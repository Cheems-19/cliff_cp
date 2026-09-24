#!/usr/bin/env python
"""Phase 16 分析：d_nn_sim 为何净加宽、d_nb 为何净收紧。

输入：results/phase16_local/dump/*.csv.gz（phase16_signal_residual.py 的产物，需先拉回本地）
输出：results/phase16_local/ANALYSIS.txt

三个统计量（都在**校准侧**计算，避免用测试标签）：
  S1  信号-残差秩相关：Spearman(signal, |y−pred|)，按 (数据集, 种子) 算再聚合
      —— 预测：d_nn_sim > 0（高相似 ≠ 易预测），d_nb < 0（邻居多 = 稠密易预测）
  S2  悬崖所在箱的残差构成：按校准侧分位把全体分子分 5 箱，悬崖分子落入的箱里
      校准分子的平均残差 / 全体校准分子平均残差（比值 > 1 = 该箱尾部重 → q 大 → 加宽）
      —— 预测：d_nn_sim 比值 > 1，d_nb 比值 < 1
  S3  同群性：大残差分子（校准侧前 10%）与悬崖分子的信号水平差（以信号 SD 为单位）
      —— 预测：d_nn_sim 差距小（同群），d_nb 差距大（不同群）
"""
from __future__ import annotations

import glob
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
DUMP = sys.argv[1] if len(sys.argv) > 1 else "results/phase16_local/dump"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/phase16_local/ANALYSIS.txt"
SIGNALS = ["d_wstd", "d_std", "d_iqr", "d_nb", "d_nn_sim", "rf_std"]
NBINS = 5

lines = []


def w(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)
    print(s, flush=True)


fs = sorted(glob.glob(os.path.join(DUMP, "*.csv.gz")))
if not fs:
    raise SystemExit(f"no dump files under {DUMP}")
fr = [pd.read_csv(f) for f in fs]
A = pd.concat(fr, ignore_index=True)
w("=" * 78)
w("Phase 16 —— 信号-残差构成分析（n=", len(A), "条分子记录）")
w(f"数据集 {A.dataset.nunique()} | 种子 {A.seed.nunique()} | cal {int((A.role=='cal').sum()):,} / "
  f"test {int((A.role=='test').sum()):,}")
w("=" * 78)

cal = A[A.role == "cal"].copy()
w("\n" + "─" * 78)
w("S1  信号-残差秩相关（校准侧；按 (数据集, 种子) 计算后聚合）")
w("─" * 78)
w(f"{'信号':12s} {'平均 rho':>9s} {'>0 的单元':>10s} {'Wilcoxon p':>12s}   读法")
per_signal = {}
for sn in SIGNALS:
    rhos = []
    for (ds, sd), g in cal.groupby(["dataset", "seed"]):
        g = g[np.isfinite(g[sn]) & np.isfinite(g.residual)]
        if len(g) < 100 or g[sn].nunique() < 5:
            continue
        rhos.append(stats.spearmanr(g[sn], g.residual).statistic)
    r = np.array(rhos)
    wl = stats.wilcoxon(r, zero_method="wilcox")
    per_signal[sn] = r.mean()
    read = "高信号=大残差（重尾同箱）" if r.mean() > 0 else "高信号=小残差（轻尾同箱）"
    w(f"{sn:12s} {r.mean():>+9.3f} {int((r > 0).sum()):>5d}/{len(r):<4d} {wl.pvalue:>12.3g}   {read}")

w("\n" + "─" * 78)
w("S2  悬崖所在箱的残差构成（箱 = 按校准侧信号分位的 5 箱；比值 >1 → 该箱 q 大 → 悬崖被加宽）")
w("─" * 78)
w(f"{'信号':12s} {'比值(均值)':>10s} {'>1 的单元':>10s} {'Wilcoxon p':>12s}   读法")
for sn in SIGNALS:
    ratios = []
    for (ds, sd), g in A.groupby(["dataset", "seed"]):
        gcal = g[g.role == "cal"].copy()
        gtest = g[g.role == "test"].copy()
        gcal = gcal[np.isfinite(gcal[sn])]
        gtest = gtest[np.isfinite(gtest[sn])]
        if len(gcal) < 200 or gcal[sn].nunique() < 5 or gtest.cliffB.sum() < 5:
            continue
        sig_all = np.where(np.isfinite(g[sn]), np.nan_to_num(g[sn], nan=0.0), 0.0)
        edges = np.quantile(sig_all[g.role == "cal"], [i / NBINS for i in range(1, NBINS)])
        gcal["bin"] = np.searchsorted(edges, np.where(np.isfinite(gcal[sn]), gcal[sn], 0.0))
        gtest["bin"] = np.searchsorted(edges, np.where(np.isfinite(gtest[sn]), gtest[sn], 0.0))
        cliff_bins = gtest[gtest.cliffB].bin.value_counts()
        if not len(cliff_bins):
            continue
        top_bin = int(cliff_bins.idxmax())          # 悬崖最集中的箱
        mu_bin = gcal[gcal.bin == top_bin].residual.mean()
        mu_all = gcal.residual.mean()
        if np.isfinite(mu_bin) and mu_all > 0:
            ratios.append(mu_bin / mu_all)
    r = np.array(ratios)
    wl = stats.wilcoxon(r - 1.0, zero_method="wilcox")
    read = "悬崖箱残差重 → 加宽" if r.mean() > 1 else "悬崖箱残差轻 → 收紧"
    w(f"{sn:12s} {r.mean():>10.3f} {int((r > 1).sum()):>5d}/{len(r):<4d} {wl.pvalue:>12.3g}   {read}")

w("\n" + "─" * 78)
w("S3  同群性：大残差分子（校准侧前 10%）与悬崖分子的信号差（以信号 SD 为单位；0=同群）")
w("─" * 78)
w(f"{'信号':12s} {'|差距| 均值':>10s}   读法")
for sn in SIGNALS:
    gaps = []
    for (ds, sd), g in cal.groupby(["dataset", "seed"]):
        g = g[np.isfinite(g[sn]) & np.isfinite(g.residual)]
        if len(g) < 200 or g[sn].nunique() < 5:
            continue
        sdv = g[sn].std()
        if not sdv or not np.isfinite(sdv):
            continue
        thr = g.residual.quantile(0.9)
        big = g[g.residual >= thr]
        cliff = g[g.cliffB]
        if not len(big) or not len(cliff):
            continue
        gaps.append(abs(big[sn].mean() - cliff[sn].mean()) / sdv)
    r = np.array(gaps)
    w(f"{sn:12s} {r.mean():>10.3f}   "
      f"{'悬崖与大残差近似同群' if r.mean() < 0.3 else '悬崖与大残差分群'}")

w("\n" + "=" * 78)
w("判读：若 S1 显示 d_nn_sim 与残差正相关（或显著弱于 d_nb 的负相关），且 S2 显示")
w("d_nn_sim 的悬崖箱残差比值 > 1 而 d_nb < 1，则「错配量判类别、net_mag 判方向」的")
w("机制解释成立：d_nn_sim 的高分箱保留了悬崖自身的重残差尾部，因此加宽而非收紧。")
w("=" * 78)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
io.open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"\n[saved] {OUT}")
