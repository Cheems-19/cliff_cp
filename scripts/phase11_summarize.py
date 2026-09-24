# -*- coding: utf-8 -*-
"""Phase 11 汇总：random vs scaffold 划分 × 方法（含 adaptive / weighted 基线）。

读数：
  1. global 的悬崖覆盖差距（gapA_pp / gapB_pp）在两种划分下是否成立（逐数据集配对 + 方向一致性）
  2. 各条件化方法相对 global 的 Δ 悬崖覆盖（配对 Wilcoxon），两种划分分开报
  3. 错配量（misalignA/B）在 scaffold 下是否仍成立
输出：results/phase11_shift/SUMMARY.txt + PAIRED.csv + SIGNALS_AGG.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/phase11_shift"
M = pd.read_csv(os.path.join(OUT, "METHODS.csv"))
S = pd.read_csv(os.path.join(OUT, "SIGNALS.csv")) if os.path.exists(os.path.join(OUT, "SIGNALS.csv")) else None

lines = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


def paired_by_dataset(df, col, base_col=None):
    """先在各数据集内跨种子求均值，再跨数据集配对。"""
    g = df.groupby(["dataset", "method"])[col].mean().unstack("method")
    return g


p("=" * 78)
p("Phase 11 汇总：划分现实性 × 基线完整性")
p("=" * 78)
p("rows:", len(M), "| datasets:", M.dataset.nunique(), "| splits:", sorted(M.split.unique()),
  "| seeds:", M.seed.nunique(), "| methods:", sorted(M.method.unique()))
p("")

# ---------- 1. global 的悬崖差距：random vs scaffold ----------
p("### 1. global 的悬崖覆盖差距（非悬崖 - 悬崖，pp）")
for split in sorted(M.split.unique()):
    sub = M[(M.split == split) & (M.method == "global")]
    for col, lab in (("gapA_pp", "cliffA(数据集级)"), ("gapB_pp", "cliffB(机制)")):
        v = sub.groupby("dataset")[col].mean().dropna()
        if len(v) == 0:
            continue
        npos = int((v > 0).sum())
        try:
            w = stats.wilcoxon(v, alternative="greater")
            pv = w.pvalue
        except Exception:
            pv = np.nan
        p(f"  [{split:8s}] {lab:16s} 合并={v.mean():+.1f}pp  "
          f"方向一致={npos}/{len(v)}  Wilcoxon p={pv:.2e}")
p("")

# ---------- 2. 各方法相对 global 的 Δ 悬崖覆盖（按划分分开） ----------
p("### 2. 条件化方法 vs global：Δ 悬崖覆盖（pp，正=改善悬崖组）")
rows = []
for split in sorted(M.split.unique()):
    sub = M[M.split == split]
    for metric, lab in (("cov_cliffA", "cliffA"), ("cov_cliffB", "cliffB")):
        piv = paired_by_dataset(sub, metric)
        if "global" not in piv.columns:
            continue
        for m in piv.columns:
            if m == "global":
                continue
            d = (piv[m] - piv["global"]).dropna()
            if len(d) == 0:
                continue
            try:
                pv = stats.wilcoxon(d).pvalue
            except Exception:
                pv = np.nan
            rows.append(dict(split=split, metric=lab, method=m,
                             delta_pp=100 * d.mean(), n_datasets=len(d),
                             n_pos=int((d > 0).sum()), wilcoxon_p=pv))
P = pd.DataFrame(rows)
P.to_csv(os.path.join(OUT, "PAIRED.csv"), index=False)
for split in sorted(P.split.unique()):
    p(f"  --- {split} ---")
    for _, r in P[P.split == split].sort_values(["metric", "delta_pp"], ascending=[True, False]).iterrows():
        p(f"    {r.metric:6s} {r.method:16s} Δ={r.delta_pp:+6.2f}pp  "
          f"({r.n_pos}/{r.n_datasets})  p={r.wilcoxon_p:.2e}")
p("")

# ---------- 3. 错配量 ----------
if S is not None and len(S):
    p("### 3. 错配量（信号对失效 AUC - 对悬崖 AUC）")
    agg = []
    for split in sorted(S.split.unique()):
        for sig, g in S[S.split == split].groupby("signal"):
            agg.append(dict(split=split, signal=sig,
                            auc_fail=g.auc_fail.mean(), auc_cliffB=g.auc_cliffB.mean(),
                            auc_cliffA=g.auc_cliffA.mean(),
                            misalignB=g.misalignB.mean(), misalignA=g.misalignA.mean()))
    A = pd.DataFrame(agg)
    A.to_csv(os.path.join(OUT, "SIGNALS_AGG.csv"), index=False)
    for split in sorted(A.split.unique()):
        p(f"  --- {split} ---")
        for _, r in A[A.split == split].iterrows():
            p(f"    {r.signal:9s} AUC_fail={r.auc_fail:.3f}  AUC_cliffB={r.auc_cliffB:.3f}  "
              f"AUC_cliffA={r.auc_cliffA:.3f}  M_B={r.misalignB:+.3f}  M_A={r.misalignA:+.3f}")

p("")
p("=" * 78)

with open(os.path.join(OUT, "SUMMARY.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("[saved]", os.path.join(OUT, "SUMMARY.txt"), flush=True)
