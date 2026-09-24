# -*- coding: utf-8 -*-
"""计算表 4 缺失的两行（Mondrian←d_wstd / d_nb），并用 rfstd 行做交叉校验。"""
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


mn = pd.read_csv(os.path.join(L, "phase11_mondrian", "METHODS.csv"))
base = pd.read_csv(os.path.join(L, "phase11_shift", "METHODS.csv"))
p("mondrian rerun rows:", len(mn), "| datasets:", mn.dataset.nunique(),
  "| splits:", sorted(mn.split.unique()), "| methods:", sorted(mn.method.unique()))

def report(df, tag):
    p(f"\n===== {tag} =====")
    for side in ["cliffA", "cliffB"]:
        col = f"cov_{side}"
        piv = df.pivot_table(index=["split", "dataset", "seed"], columns="method", values=col)
        for split in ["random", "scaffold"]:
            if split not in piv.index.get_level_values("split"):
                continue
            sub = piv.xs(split, level="split")
            g = sub["global"]
            p(f"  --- {side} / {split} ---")
            rows = []
            for meth in sorted(c for c in sub.columns if c != "global"):
                d = (sub[meth] - g).dropna()
                d_ds = d.groupby(level=0).mean()
                try:
                    pv = float(stats.wilcoxon(d_ds, zero_method="wilcox").pvalue)
                except Exception:
                    pv = float("nan")
                rows.append((meth, 100 * d_ds.mean(), int((d_ds > 0).sum()), len(d_ds), pv))
            for meth, delta, npos, ntot, pv in sorted(rows, key=lambda r: -r[1]):
                p(f"    {meth:16s} Δ={delta:+6.2f}pp  ({npos:2d}/{ntot})  p={pv:.2e}")
    # 宽度
    p(f"  --- 宽度（相对 global %） ---")
    for split in ["random", "scaffold"]:
        s = df[df.split == split]
        w = s.groupby("method").width_all.mean()
        b = w["global"]
        p("   ", split, {m: f"{(w[m]/b-1)*100:+.2f}%" for m in sorted(w.index)})


report(mn, "补跑（修复后）")
report(base, "首跑（phase11_shift，d_wstd 臂有 bug）")

# 一致性：rfstd 行应与首跑一致
pa = mn[mn.method == "mondrian_rfstd"].set_index(["split", "dataset", "seed"])["cov_cliffA"]
pb = base[base.method == "mondrian_rfstd"].set_index(["split", "dataset", "seed"])["cov_cliffA"]
common = pa.index.intersection(pb.index)
diff = (pa.loc[common] - pb.loc[common]).abs()
p(f"\n### 交叉校验 mondrian_rfstd（补跑 vs 首跑）")
p(f"  可比对 (split,dataset,seed) 数 = {len(common)}  最大绝对差 = {diff.max():.3e}  均值差 = {diff.mean():.3e}")
p("  → 若≈0，说明补跑与首跑在同一划分/种子上可复现，两行数字可拼接进同一张表")

io.open(os.path.join(L, "PHASE11_MONDRIAN_判读.txt"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")
print("\n[saved] results_server/PHASE11_MONDRIAN_判读.txt")
