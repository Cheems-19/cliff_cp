# -*- coding: utf-8 -*-
"""判读 phase11 / phase13 结果，产出 §3.8 表 4/表 5 所需的确定数字。

R1 = 骨架划分下 P1 是否仍成立
R2 = adaptive/weighted 是否消除差距
R3 = 固定预测器下，信号精度与伤害是否单调
"""
import io, os
import numpy as np
import pandas as pd
from scipy import stats

L = r"C:/Users/Administrator/Desktop/小论文/cliff_cp/results_server"
OUT = []


def p(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


# ═══════════════ R1 / R2：phase11 ═══════════════
m = pd.read_csv(os.path.join(L, "phase11_shift", "METHODS.csv"))
p("### METHODS.csv columns:", list(m.columns))
p("rows:", len(m), "| datasets:", m.dataset.nunique(),
  "| splits:", sorted(m.split.unique()), "| methods:", sorted(m.method.unique()))

# 逐 (dataset, seed) 求各方法相对 global 的差，再跨数据集配对
key = ["dataset", "seed"]
for side in ["cliffA", "cliffB"]:
    col = f"cov_{side}"
    if col not in m.columns:
        continue
    p(f"\n### 方法 vs global —— {side} 覆盖率（α=0.10）")
    piv = m.pivot_table(index=["split"] + key, columns="method", values=col)
    for split in ["random", "scaffold"]:
        sub = piv.xs(split, level="split")
        if "global" not in sub.columns:
            continue
        g = sub["global"]
        rows = []
        for meth in sorted([c for c in sub.columns if c != "global"]):
            d = (sub[meth] - g).dropna()
            if len(d) == 0:
                continue
            # 先在数据集内跨种子取均值，再跨数据集检验
            d_ds = d.groupby(level=0).mean()
            try:
                w = stats.wilcoxon(d_ds, zero_method="wilcox")
                pv = float(w.pvalue)
            except Exception:
                pv = float("nan")
            rows.append((meth, 100 * d_ds.mean(), int((d_ds > 0).sum()), len(d_ds), pv))
        rows.sort(key=lambda r: -r[1])
        p(f"  --- {split} ---")
        for meth, delta, npos, ntot, pv in rows:
            p(f"    {meth:16s} Δ={delta:+6.2f}pp  ({npos:2d}/{ntot})  p={pv:.2e}")

# ═══════════════ R3：phase13 ═══════════════
ld = pd.read_csv(os.path.join(L, "phase13", "LADDER.csv"))
p("\n### LADDER.csv columns:", list(ld.columns))
p("rows:", len(ld), "| datasets:", ld.dataset.nunique(), "| seeds:", ld.seed.nunique(),
  "| k values:", sorted(ld.k.unique()))

agg = ld.groupby("k").agg(
    AUC_fail=("auc_fail", "mean") if "auc_fail" in ld.columns else ("AUC_fail", "mean"),
).reset_index()
p("\n### k 阶梯（跨 15 数据集 × 5 种子均值）")
cand = {c.lower(): c for c in ld.columns}
def pick(*names):
    for n in names:
        if n in ld.columns:
            return n
    return None
c_fail = pick("auc_fail")
c_cliffB = pick("auc_cliffB", "auc_cliffb")
c_mis = pick("misalign", "M")
c_dcB = pick("d_cov_cliffB_pp", "dcovB_pp", "dCovB_pp", "dCovB")
c_dcA = pick("d_cov_cliffA_pp", "dcovA_pp", "dCovA_pp", "dCovA")
p("resolved cols:", c_fail, c_cliffB, c_mis, c_dcB, c_dcA)

tab = ld.groupby("k").mean(numeric_only=True)
for k, r in tab.iterrows():
    p(f"  k={int(k):3d}  AUC_fail={r[c_fail]:.3f}  AUC_cliffB={(r[c_cliffB] if c_cliffB else float('nan')):.3f}"
      f"  M={r[c_mis]:+.3f}  dCovB={r[c_dcB]:+.2f}pp  dCovA={r[c_dcA]:+.2f}pp "
      f" harm<0={int((ld[ld.k==k][c_dcB] < 0).sum())}/{ld.dataset.nunique()}")

ks = sorted(ld.k.unique())
rho_fail = stats.spearmanr(ld[c_fail], ld.k)
rho_mis = stats.spearmanr(ld[c_mis], ld.k)
rho_dc = stats.spearmanr(ld[c_dcB].abs(), ld.k)
p("\n### 单调性（逐行，n=%d）" % len(ld))
p(f"  Spearman(AUC_fail, k)   = {rho_fail.statistic:+.3f}  p={rho_fail.pvalue:.2e}")
p(f"  Spearman(|M|, k)        = {rho_mis.statistic:+.3f}  p={rho_mis.pvalue:.2e}")
p(f"  Spearman(|dCovB|, k)    = {rho_dc.statistic:+.3f}  p={rho_dc.pvalue:.2e}")

# spearman on the aggregated means (7 points) too
agg2 = ld.groupby("k").mean(numeric_only=True)
p(f"  Spearman(AUC_fail_mean, k) = {stats.spearmanr(agg2.index, agg2[c_fail]).statistic:+.3f}")
p(f"  Spearman(|dCovB_mean|, k)  = {stats.spearmanr(agg2.index, agg2[c_dcB].abs()).statistic:+.3f}")

# 大 k vs 小 k 的配对检验（固定预测器下的剂量效应）
lo = ld[ld.k.isin([2, 5])].groupby(["dataset", "seed"])[c_dcB].mean()
hi = ld[ld.k.isin([100, 200])].groupby(["dataset", "seed"])[c_dcB].mean()
dd = (hi - lo).dropna()
w = stats.wilcoxon(dd, zero_method="wilcox")
p(f"\n### 剂量效应：dCovB(k=100,200) − dCovB(k=2,5)")
p(f"  均值差 = {dd.mean():+.2f} pp  (n={len(dd)})  Wilcoxon p={w.pvalue:.2e}"
  f"  方向为负 {int((dd<0).sum())}/{len(dd)}")

# k=200 是否显著伤害
d200 = ld[ld.k == 200].groupby("dataset")[c_dcB].mean()
w200 = stats.wilcoxon(d200, zero_method="wilcox")
p(f"\n### k=200（全部树，信号最充分）")
p(f"  均值 = {d200.mean():+.2f} pp  (n={len(d200)})  Wilcoxon p={w200.pvalue:.2e}"
  f"  方向为负 {int((d200<0).sum())}/{len(d200)}")

io.open(r"C:/Users/Administrator/Desktop/小论文/cliff_cp/results_server/PHASE11_13_判读.txt",
        "w", encoding="utf-8").write("\n".join(OUT) + "\n")
print("\n[saved] results_server/PHASE11_13_判读.txt")
