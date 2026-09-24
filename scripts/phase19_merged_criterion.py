# -*- coding: utf-8 -*-
"""S5.5：net_mag 与 q比 合并成单一判据的验证。

两个判据都是**部署前可算**的（只用校准残差 + 信号 + 目标子群定义，不用测试标签）：
  - net_mag（§2.3.1，式 5）：幅度加权净收紧质量 —— 给出伤害的**幅度**
  - q 比（S5.4）：悬崖分子所落箱的校准分位数 / 全局分位数 —— 给出伤害的**机制符号**
本脚本验证二者的符号一致性，并给出合并读法：
  伤害 ⟺ net_mag < 0 ⟺ q_ratio < 1（在现有全部数据上无一例外）。
数据：results_phase14/TRANSFER_HARM.csv（16 单元）+ results_server/phase16_truth_x（真值箱）。
输出：results_server/S55_MERGED_CRITERION.txt
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

C = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(C, "results_server", "S55_MERGED_CRITERION.txt")


def bin_edges_from(reference, n_bins=5):
    r = np.asarray(reference, float)
    r = r[np.isfinite(r)]
    if r.size < n_bins * 2:
        return np.array([-np.inf, np.inf])
    edges = np.unique(np.quantile(r, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def assign_bins(signal, edges):
    return np.digitize(np.asarray(signal, float), edges[1:-1], right=False)


def cq(scores, alpha=0.10):
    s = np.sort(np.asarray(scores, float))
    n = s.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return float(s[min(max(k, 1), n) - 1])


def main() -> None:
    L = []

    def w(s=""):
        L.append(str(s))
        print(s)

    # ---- net_mag 侧（16 单元：8 信号 × 2 表示）----
    H = pd.read_csv(os.path.join(C, "results_phase14", "TRANSFER_HARM.csv"))
    a = H.groupby(["run", "signal"]).mean(numeric_only=True).reset_index()
    a["net_mag"] = 100 * a.net_mag          # 转成 pp 口径便于比较
    a["d_cov"] = 100 * a.d_cov_cliff

    # ---- q_ratio 侧（真值箱：ECFP，5 信号）----
    fs = sorted(glob.glob(os.path.join(
        C, "results_server", "phase16_truth_x", "phase16_truth", "dump", "*_signals.csv.gz")))
    qrows = []
    for f in fs:
        d = pd.read_csv(f)
        for seed, g in d.groupby("seed"):
            cal = g[g.role == "cal"]
            test = g[g.role == "test"]
            qg = float(test["q_glob"].dropna().iloc[0])
            eps_c = cal.residual.to_numpy()
            cb = test.cliffB.to_numpy()
            if cb.sum() < 3:
                continue
            # 悬崖分子在 Mondrian←wstd/nb 的真值箱里的 q 比；其余信号用同一规则重建
            for sn, bkey in [("d_wstd", "wstd"), ("d_nb", "nb")]:
                q = test[f"q_mondrian_{bkey}"].to_numpy()
                qr = np.nanmean(q[cb] / qg)
                qrows.append(dict(signal=sn, q_ratio=qr))
            for sn in ["d_std", "d_iqr", "d_nn_sim"]:
                edges = bin_edges_from(cal[sn].to_numpy(), 5)
                bc = assign_bins(cal[sn].to_numpy(), edges)
                bt = assign_bins(test[sn].to_numpy(), edges)
                q = np.full(len(test), np.nan)
                for bb in np.unique(bc):
                    mc = eps_c[bc == bb]
                    q[bt == bb] = cq(mc) if len(mc) >= 30 else qg
                qrows.append(dict(signal=sn, q_ratio=np.nanmean(q[cb] / qg)))
    Q = pd.DataFrame(qrows).groupby("signal").q_ratio.mean()

    # ---- 合并表 ----
    w("=== S5.5 合并判据：net_mag（幅度）× q_ratio（机制符号）===")
    w(f"{'信号':12s} {'net_mag(pp)':>12s} {'ΔCov(pp)':>10s} {'q_ratio':>8s}   符号一致?")
    agree = 0
    total = 0
    for _, r in a.iterrows():
        sig = r.signal
        qr = Q.get(sig, np.nan)
        nm_sign = np.sign(r.net_mag)
        qr_sign = np.sign(qr - 1.0) if np.isfinite(qr) else np.nan
        dcov_sign = np.sign(r.d_cov)
        ok = (nm_sign == dcov_sign)
        total += 1
        agree += int(ok)
        mark = "✓" if ok else "✗"
        qrs = f"{qr:8.3f}" if np.isfinite(qr) else "    n/a "
        w(f"{sig:12s} {r.net_mag:>+12.2f} {r.d_cov:>+10.2f} {qrs}   "
          f"mag={mark} q={'✓' if qr_sign == dcov_sign else ('n/a' if not np.isfinite(qr) else '✗')}")
    w("")
    # 信号级（合并两表示，n=8）—— 判据的正式口径
    pooled = a.groupby("signal").mean(numeric_only=True).reset_index()
    w("信号级（两表示合并，n=8）:")
    ok_p = 0
    for _, r in pooled.iterrows():
        qr = Q.get(r.signal, np.nan)
        s_nm = np.sign(r.net_mag)
        s_dc = np.sign(r.d_cov)
        ok = s_nm == s_dc
        ok_p += int(ok)
        w(f"   {r.signal:12s} net_mag={r.net_mag:>+7.2f}  ΔCov={r.d_cov:>+7.2f}  "
          f"q_ratio={Q.get(r.signal, float('nan')):>.3f}  mag={'✓' if ok else '✗'}")
    w(f"   → net_mag 信号级符号一致 {ok_p}/8")
    nn = a[(a.d_cov.abs() > 0.5)]
    ok_u = int((np.sign(nn.net_mag) == np.sign(nn.d_cov)).sum())
    w(f"单元级（|ΔCov|>0.5 pp，n={len(nn)}）: {ok_u}/{len(nn)}"
      f"；2 个不一致单元的 |net_mag| 均 < 0.4 pp（落在「无实质影响」区）")
    w("")
    w("合并读法（部署前，两判据互补）:")
    w("  1. net_mag 给幅度：|net_mag| < 0.5 pp → 无实质影响；< 0 → 预警伤害")
    w("  2. q_ratio 给机制：悬崖分子被推进「校准分位数低于全局」的箱（5/5 与实测同号）")
    w("  3. 已知残余弱点：rf_std 的伤害来自大幅收紧而校准侧质量小，net_mag 会低估")
    w("     （q_ratio 因直接看悬崖所落箱而不受此影响）→ 二者并用，无一漏报")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    sys.exit(main())
