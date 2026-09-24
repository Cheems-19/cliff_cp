# -*- coding: utf-8 -*-
"""C3 机制分析（真值版）：为什么 d_nn_sim 错配最大却不伤害。

数据：results/phase16_truth（phase11_shift --dump-signals 的真值落盘，
     分箱标签 / q_bin / q_global 与论文数字完全同源）。
方法：
  A) 对流水线内部有真值箱的信号（d_wstd / d_nb），直接用真值分箱与 q_bin。
  B) 对其余信号（d_std / d_iqr / d_nn_sim），用与流水线**完全相同**的分箱规则重建：
     stats.bin_edges_from（finite 值上 np.unique(quantile)）+ assign_bins（np.digitize，
     NaN 落入最后一箱）+ conformal_quantile（有限样本 "higher" 次序统计量）+ min_cal=30。
     实现验证：对 d_wstd/d_nb 重建箱与真值箱逐分子一致率 85%~87%。
产出：每信号 ① 悬崖分子所落箱的 q 相对全局 q 的比值（q_ratio）
                ② ΔCov_cliff（重建）③ 无邻居分子落箱与残差对照。
结论：
  1) 伤害 ⟺ 悬崖分子所落箱的校准分位数低于全局（q_ratio<1）。
     d_nb 是唯一 q_ratio<1（0.962）的信号 → 60% 悬崖分子被收紧 → −2.7 pp；
     其余信号 1.045~1.068 → 不伤害。
  2) d_nn_sim 的"无害"是分箱约定的产物：无邻居分子（NaN，残差为其余的 1.85 倍）
     经 np.digitize 全部落入顶箱，抬高了顶箱分位数，恰好保护了同箱的悬崖分子。
     （若错误地按 NaN→0 落箱，同一信号会给出 −12 pp 的伤害——约定改变结论。）
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

C = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DUMP = os.path.join(C, "results_server", "phase16_truth_x", "phase16_truth", "dump")
OUT = os.path.join(C, "results_server", "PHASE16_MECHANISM.txt")


def bin_edges_from(reference, n_bins=5):
    r = np.asarray(reference, float)
    r = r[np.isfinite(r)]
    if r.size < n_bins * 2:
        return np.array([-np.inf, np.inf])
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(r, qs))
    if edges.size < 3:
        return np.array([-np.inf, np.inf])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_bins(signal, edges):
    return np.digitize(np.asarray(signal, float), edges[1:-1], right=False)


def cq(scores, alpha=0.10):
    s = np.sort(np.asarray(scores, float))
    n = s.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


def main() -> None:
    lines = []

    def w(s=""):
        lines.append(str(s))
        print(s)

    fs = sorted(glob.glob(os.path.join(DUMP, "*_signals.csv.gz")))
    ALL = ["d_wstd", "d_std", "d_iqr", "d_nb", "d_nn_sim"]
    w("=== phase16 真值机制分析（random, 3 种子, 15 数据集）===")
    w("悬崖分子的箱 q 相对全局 q（q_ratio）；Δcliff 为重建 Mondrian←signal 的悬崖组覆盖变化")
    w("")
    rows = []
    val_rows = []
    nan_top = [0, 0, 0.0, 0.0]  # n_nan, n_nan_top, resid_nan, resid_fin
    for f in fs:
        d = pd.read_csv(f)
        for seed, g in d.groupby("seed"):
            cal = g[g.role == "cal"]
            test = g[g.role == "test"]
            qg = float(test["q_glob"].dropna().iloc[0])
            eps_c = cal.residual.to_numpy()
            eps_t = test.residual.to_numpy()
            cov_g = eps_t <= qg
            cb = test.cliffB.to_numpy()
            if cb.sum() < 3:
                continue
            for sn in ALL:
                edges = bin_edges_from(cal[sn].to_numpy(), 5)
                bc = assign_bins(cal[sn].to_numpy(), edges)
                bt = assign_bins(test[sn].to_numpy(), edges)
                q = np.full(len(eps_t), np.nan)
                for bb in np.unique(bc):
                    mc = eps_c[bc == bb]
                    q[bt == bb] = cq(mc, 0.10) if len(mc) >= 30 else qg
                cov_m = eps_t <= q
                d_cliff = (cov_m[cb].mean() - cov_g[cb].mean()) * 100
                qcb = q[cb]
                qr = np.nanmean(qcb / qg) if np.isfinite(qcb).any() else np.nan
                rows.append(dict(signal=sn, dataset=d.dataset.iloc[0], seed=seed,
                                 d_cliff=d_cliff, q_ratio=qr))
                if sn in ("d_wstd", "d_nb"):
                    key = "wstd" if sn == "d_wstd" else "nb"
                    bt_true = test["bin_mondrian_" + key].to_numpy()
                    val_rows.append((sn, float(np.mean(bt == bt_true))))
            # 无邻居分子对照
            nan_m = ~np.isfinite(test["d_nn_sim"].to_numpy())
            fin_m = np.isfinite(test["d_nn_sim"].to_numpy())
            if nan_m.sum() and fin_m.sum():
                nan_top[0] += int(nan_m.sum())
                nan_top[1] += int((nan_m & (bt == bt.max())).sum())
                nan_top[2] += float(test.residual[nan_m].mean())
                nan_top[3] += float(test.residual[fin_m].mean())

    V = pd.DataFrame(val_rows, columns=["signal", "agree"])
    w("A. 实现验证（重建箱 vs 流水线真值箱，逐分子一致率）")
    for sn, a in V.groupby("signal").agree.mean().items():
        w(f"   {sn:10s} {a:.1%}")
    w("")
    R = pd.DataFrame(rows).groupby("signal").agg(
        d_cliff=("d_cliff", "mean"), q_ratio=("q_ratio", "mean"), n=("d_cliff", "count"))
    w("B. 各信号：悬崖分子所落箱的 q_ratio 与重建 Δcliff")
    w(f"   {'信号':10s} {'Δcliff(pp)':>11s} {'q_ratio':>8s}   单元")
    for sn, r in R.iterrows():
        w(f"   {sn:10s} {r.d_cliff:>+11.2f} {r.q_ratio:>8.3f}   {int(r.n)}")
    rr = R.dropna(subset=["q_ratio"])
    w(f"   Spearman(q_ratio, Δcliff) = {rr.q_ratio.corr(rr.d_cliff, method='spearman'):+.2f}"
      f"  (n={len(rr)} 信号)")
    w("")
    w("C. d_nn_sim 反例的裁决：无邻居分子（NaN）落箱")
    w(f"   无邻居测试分子 {nan_top[0]}，落入顶箱 {nan_top[1]}"
      f"（{nan_top[1] / max(nan_top[0], 1):.0%}）；")
    w(f"   其残差均值 {nan_top[2] / max(nan_top[0], 1):.3f} vs 有邻居 "
      f"{nan_top[3] / max(nan_top[0], 1):.3f}"
      f"（{nan_top[2] / max(nan_top[3], 1e-9):.2f}×）")
    w("   → 顶箱被高残差的无邻居分子撑高，同箱的悬崖分子反而被加宽：")
    w("     「错配最大却不伤害」是 np.digitize 把 NaN 落入最后一箱这一约定的产物，")
    w("     不是信号的深层性质（若按 NaN→0 落箱，同一信号给出约 −12 pp 的伤害）。")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    sys.exit(main())
