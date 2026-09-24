#!/usr/bin/env python
"""Phase 15 汇总：偏移强度谱的判读。

产出 SUMMARY.txt，含四块：
  (1) 各协议的宏观覆盖与悬崖差距（cliffA / cliffB），含方向一致性与 Wilcoxon；
  (2) 单调性检验：相邻协议的两两配对差，给出偏移强度与失效的关系；
  (3) 边际覆盖是否随偏移而跌破名义值（可交换性被打破的直接证据）；
  (4) 连续偏移强度：按测试分子到训练集的最大相似度分层，分协议组计算
      —— 回答"悬崖失效是不是只是'离训练集远'的另一种说法"。
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/phase15_spectrum"
PROTO_ORDER = ["random", "scaffold", "scaffold_gen", "act_shift40", "act_shift20"]
PROTO_LABEL = {"random": "P0 random", "scaffold": "P1 scaffold (exact)",
               "scaffold_gen": "P2 scaffold (generic)", "act_shift40": "P3 act shift 40%",
               "act_shift20": "P4 act shift 20%"}

lines: list[str] = []


def w(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)
    print(s, flush=True)


M = pd.read_csv(os.path.join(OUT, "METHODS.csv"))
protos = [p for p in PROTO_ORDER if p in set(M.protocol)]
g = M[M.method == "global"].copy()

w("=" * 78)
w("Phase 15 —— 偏移强度谱判读")
w("=" * 78)
w(f"数据集 {g.dataset.nunique()} 个 | 协议 {protos} | global 行数 {len(g)}")


def cell_rows(metric_cov_cliff, metric_cov_non, tag):
    rows = []
    for p in protos:
        s = g[g.protocol == p]
        per = s.groupby("dataset")[[metric_cov_cliff, metric_cov_non]].mean()
        per["gap"] = (per[metric_cov_non] - per[metric_cov_cliff]) * 100
        per = per[np.isfinite(per["gap"])]
        if len(per) < 3:
            continue
        wl = stats.wilcoxon(per["gap"], zero_method="wilcox")
        rows.append(dict(protocol=p, tag=tag, n=len(per), gap=per["gap"].mean(),
                         lo=per["gap"].min(), hi=per["gap"].max(),
                         pos=int((per["gap"] > 0).sum()), p=wl.pvalue,
                         cov_cliff=per[metric_cov_cliff].mean(),
                         cov_non=per[metric_cov_non].mean(),
                         per=per["gap"]))
    return rows


w("\n" + "─" * 78)
w("(1) 各协议的悬崖覆盖差距（macro，逐数据集均值）")
w("─" * 78)
w(f"{'协议':24s} {'定义':7s} {'Δ(pp)':>8s} {'范围':>18s} {'方向':>7s} {'Wilcoxon p':>12s}")
all_rows = {}
for tag, c1, c2 in (("cliffA", "cov_cliffA", "cov_noncliffA"), ("cliffB", "cov_cliffB", "cov_noncliffB")):
    rows = cell_rows(c1, c2, tag)
    all_rows[tag] = rows
    for r in rows:
        rng = "{:+.1f}~{:+.1f}".format(r["lo"], r["hi"])
        w(f"{PROTO_LABEL[r['protocol']]:24s} {tag:7s} {r['gap']:+8.1f} "
          f"{rng:>18s} {r['pos']:>3d}/{r['n']:<3d} {r['p']:>12.2e}")

w("\n" + "─" * 78)
w("(2) 单调性：相邻协议的两两配对差（同一数据集上）")
w("─" * 78)
for tag in ("cliffA", "cliffB"):
    rows = {r["protocol"]: r["per"] for r in all_rows[tag]}
    chain = [p for p in protos if p in rows]
    w(f"  [{tag}]")
    for a, b in zip(chain, chain[1:]):
        common = rows[a].index.intersection(rows[b].index)
        if len(common) < 3:
            continue
        d = (rows[b][common] - rows[a][common]).to_numpy(float)
        wl = stats.wilcoxon(d, zero_method="wilcox")
        arrow = "恶化" if d.mean() > 0 else "缓解"
        w(f"    {PROTO_LABEL[a]} -> {PROTO_LABEL[b]}: "
          f"dDelta = {d.mean():+6.1f} pp  (n={len(d)}, p={wl.pvalue:.3g}) {arrow}")

w("\n" + "─" * 78)
w("(3) 边际覆盖是否随偏移跌破名义 0.90（可交换性被打破的直接证据）")
w("─" * 78)
w(f"{'协议':24s} {'覆盖(全)':>9s} {'<0.88 的数据集':>14s} {'测试 yy 均值 - 训练 yy':>22s}")
for p in protos:
    s = g[g.protocol == p]
    per = s.groupby("dataset").cov_all.mean()
    dy = (s.y_mean_test - s.y_mean_train).mean()
    w(f"{PROTO_LABEL[p]:24s} {s.cov_all.mean():>9.4f} "
      f"{int((per < 0.88).sum()):>6d}/{len(per):<7d} {dy:>+22.2f}")

w("\n" + "─" * 78)
w("(3b) 活性偏移下有没有方法能救？各方法在 act_shift 协议的边际覆盖")
w("─" * 78)
for p in [x for x in ("act_shift40", "act_shift20") if x in protos]:
    s = M[M.protocol == p]
    piv = s.groupby("method").cov_all.mean().sort_values(ascending=False)
    w(f"  [{PROTO_LABEL[p]}]  " + "  ".join(f"{k}={v:.3f}" for k, v in piv.items()))

# (4) 连续偏移强度：按 nn_sim_train 分层（分协议组，避免地板效应污染）
dump_dir = os.path.join(OUT, "dump")
if os.path.isdir(dump_dir):
    fs = sorted(f for f in os.listdir(dump_dir) if f.endswith(".csv.gz"))
    if fs:
        w("\n" + "─" * 78)
        w("(4) 连续偏移强度：按「到训练集的最大相似度」分层")
        w("    注意：活性偏移协议下边际覆盖整体崩塌（地板效应），会把分层曲线压扁，")
        w("    故本节分协议组计算——结构协议看悬崖差距是否随相似度上升，活性偏移单列作对照。")
        w("─" * 78)
        fr = [pd.read_csv(os.path.join(dump_dir, f)) for f in fs]
        A = pd.concat(fr, ignore_index=True)
        A = A[np.isfinite(A.nn_sim_train)]
        w(f"\n  合并测试记录 = {len(A):,}")
        groups = [("P0 random（基准）", ["random"]),
                  ("结构协议合并（P0+P1+P2）", ["random", "scaffold", "scaffold_gen"]),
                  ("活性偏移合并（P3+P4，对照）", ["act_shift40", "act_shift20"])]
        for gname, plist in groups:
            B = A[A.protocol.isin(plist)].copy()
            if not len(B):
                continue
            B["decile"] = pd.qcut(B.nn_sim_train, 10, labels=False, duplicates="drop")
            w(f"\n  -- {gname}  (n={len(B):,}) --")
            w(f"  {'十分位':>6s} {'n':>8s} {'相似度中位':>10s} {'覆盖(全)':>9s} "
              f"{'覆盖(悬崖A)':>11s} {'覆盖(非悬崖A)':>13s} {'Δ(pp)':>8s}")
            for dq, sub in B.groupby("decile"):
                cl = sub[sub.cliffA]
                ncl = sub[~sub.cliffA]
                gap = (ncl.covered_global.mean() - cl.covered_global.mean()) * 100 \
                    if len(cl) and len(ncl) else np.nan
                w(f"  {int(dq) + 1:>6d} {len(sub):>8,d} {sub.nn_sim_train.median():>10.3f} "
                  f"{sub.covered_global.mean():>9.4f} "
                  f"{cl.covered_global.mean() if len(cl) else float('nan'):>11.4f} "
                  f"{ncl.covered_global.mean() if len(ncl) else float('nan'):>13.4f} "
                  f"{gap:>8.1f}")
            r = stats.spearmanr(B.nn_sim_train, B.covered_global.astype(int))
            w(f"  逐分子 Spearman(nn_sim_train, covered) = {r.statistic:+.3f} (p={r.pvalue:.2e})")
            share = B.groupby("decile").cliffA.mean() * 100
            w("  悬崖A 占比(%)：", "  ".join(f"{v:.1f}" for v in share.tolist()))

w("\n" + "=" * 78)
w("读法提示：若 (1) 的差距随协议强度单调变化、而 (4) 显示悬崖差距在高相似度分层")
w("依然存在，则说明悬崖失效不是'离训练集远'的另一种说法——这正是本文主张的核心。")
w("=" * 78)

io.open(os.path.join(OUT, "SUMMARY.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"\n[saved] {OUT}/SUMMARY.txt")
