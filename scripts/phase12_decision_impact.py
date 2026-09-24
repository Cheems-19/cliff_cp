# -*- coding: utf-8 -*-
"""P0-C 决策影响实验（v2）：把"悬崖处覆盖失效"翻译成药物筛选里最痛的一种错误——
**低估风险（deprioritization risk）**：区间上界 pred+q 低于分子的真实效力 y，
于是这个真活性分子在筛选决策里被判为"不够强"而被放弃。

对真活性分子（y >= 该数据集 y 的第 τ 分位）：
  under_rate = P(y > pred + q)   # 低估：把强分子看弱了 → 放弃真lead
  over_rate  = P(y < pred - q)   # 高估：把弱分子看强了 → 假lead
  miscoverage = P(y 落在区间外)
分悬崖 / 非悬崖报告；再比较不同分区信号（global / wstd / rfstd / nb）。

输出：results_decision/DECISION_IMPACT.csv、DECISION_IMPACT_SUMMARY.csv
"""
import os
import glob
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

DUMP = "results_ecfp_abl/dump"
OUT = "results_decision"
os.makedirs(OUT, exist_ok=True)

METHODS = {
    "global": ("covered_global", "q_global"),
    "mondrian_wstd": ("covered_wstd", "q_wstd"),
    "mondrian_rfstd": ("covered_rfstd", "q_rfstd"),
    "mondrian_nb": ("covered_nb", "q_nb"),
}
TAUS = [0.50, 0.75, 0.90]

frames = []
for p in sorted(glob.glob(os.path.join(DUMP, "*.csv.gz"))):
    d = pd.read_csv(p)
    d["dataset"] = os.path.basename(p).replace("_molecules.csv.gz", "")
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
print("total molecules:", len(df), "| datasets:", df["dataset"].nunique(),
      "| cliffB frac:", round(float(df["cliffB"].mean()), 4), flush=True)

rows = []
for ds, g in df.groupby("dataset"):
    y = g["y"].to_numpy(float)
    pred = g["pred"].to_numpy(float)
    isc = g["cliffB"].to_numpy(bool)
    for mname, (cov_col, q_col) in METHODS.items():
        q = g[q_col].to_numpy(float)
        up = pred + q
        lo = pred - q
        for tau_q in TAUS:
            thr = float(np.quantile(y, tau_q))
            potent = y >= thr
            for group, gmask in (("cliff", isc), ("noncliff", ~isc), ("all", np.ones_like(isc))):
                sel = gmask & potent
                n = int(sel.sum())
                if n == 0:
                    continue
                rows.append(dict(
                    dataset=ds, method=mname, tau_q=tau_q, group=group, n=n,
                    under_rate=float((y[sel] > up[sel]).mean()),
                    over_rate=float((y[sel] < lo[sel]).mean()),
                    miscoverage=float((~((y[sel] >= lo[sel]) & (y[sel] <= up[sel]))).mean()),
                    width=float((up[sel] - lo[sel]).mean()),
                ))

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "DECISION_IMPACT.csv"), index=False)

def paired(g, col, a="cliff", b="noncliff"):
    ga = g[g.group == a].set_index("dataset")[col]
    gb = g[g.group == b].set_index("dataset")[col]
    common = ga.index.intersection(gb.index)
    d = (ga.loc[common] - gb.loc[common]).to_numpy(float)
    return float(ga.mean()), float(gb.mean()), float(100*d.mean()), int((d > 0).sum()), int(len(d))

summ = []
for (mname, tau_q), g in R.groupby(["method", "tau_q"]):
    u_c, u_n, u_gap, u_pos, nD = paired(g, "under_rate")
    m_c, m_n, m_gap, m_pos, _ = paired(g, "miscoverage")
    summ.append(dict(method=mname, tau_q=tau_q,
                     under_cliff=u_c, under_noncliff=u_n, under_gap_pp=u_gap, under_pos=f"{u_pos}/{nD}",
                     miscov_cliff=m_c, miscov_noncliff=m_n, miscov_gap_pp=m_gap,
                     ratio=f"{(u_c/u_n if u_n>0 else np.nan):.2f}x"))
S = pd.DataFrame(summ).sort_values(["tau_q", "method"])
S.to_csv(os.path.join(OUT, "DECISION_IMPACT_SUMMARY.csv"), index=False)

with pd.option_context("display.width", 240, "display.max_columns", 30):
    print("\n=== 低估风险 under_rate：真活性分子中，区间上界低于真实效力的比例 ===")
    print(S.to_string(index=False), flush=True)

print("\n[HEADLINE tau=0.75] 真活性分子被低估（淘汰）的风险：")
for m in METHODS:
    r = S[(S.method == m) & (S.tau_q == 0.75)]
    if len(r):
        r = r.iloc[0]
        print(f"  {m:16s} cliff={r.under_cliff*100:5.1f}%  non-cliff={r.under_noncliff*100:5.1f}%  "
              f"差={r.under_gap_pp:+5.1f}pp  比值={r.ratio}", flush=True)
