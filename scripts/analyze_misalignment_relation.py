#!/usr/bin/env python
"""分析：错配量与伤害的关系（含/不含已知反例 nnsim）。"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

FAMILIES = [
    ("RF+ECFP", "results_phase2/C_misalignment.csv", "results_ablation/PHASE1_ABLATION.csv"),
    ("GBR+ECFP", "results_xa_gbr/C_misalignment.csv", "results_xa_gbr/PHASE1_ABLATION.csv"),
    ("D-MPNN+ECFP", "results_xa_gnn/C_misalignment.csv", "results_xa_gnn/PHASE1_ABLATION.csv"),
    ("RF+RDKit", "results_phase2/C_misalignment.csv", "results_xa_rdkit/PHASE1_ABLATION.csv"),
]
SIG2METHOD = {"d_wstd": "wstd", "d_std": "std", "d_iqr": "iqr", "d_nb": "nb",
              "d_nn_sim": "nnsim", "d_pair_gap": "pair", "rf_std": "rfstd"}

rows = []
for fam, fmis, fabl in FAMILIES:
    mis = pd.read_csv(fmis)
    abl = pd.read_csv(fabl)
    m = dict(zip(mis["signal"], mis["gap_predict_minus_see"]))
    b = dict(zip(mis["signal"], mis["auc_predicts_fail"]))
    a = dict(zip(abl["method"].str.replace("mondrian_", ""), abl["delta_mean"] * 100))
    for sig, meth in SIG2METHOD.items():
        if sig in m and meth in a:
            rows.append({"family": fam, "signal": meth,
                         "mis": m[sig], "auc_fail": b[sig], "harm": a[meth]})
M = pd.DataFrame(rows)
print(f"n = {len(M)}")
print(f"全点        Pearson r = {np.corrcoef(M.mis, M.harm)[0,1]:+.3f}")
E = M[M.signal != "nnsim"]
print(f"排除 nnsim  Pearson r = {np.corrcoef(E.mis, E.harm)[0,1]:+.3f}  (n={len(E)})")
print(f"排除 nnsim  Spearman ρ = {pd.Series(E.mis).corr(pd.Series(E.harm), method='spearman'):+.3f}")

print("\n=== 逐信号 × 逐族 ===")
piv = M.pivot_table(index="signal", columns="family", values=["mis", "harm"])
print(piv.round(2).to_string())

print("\n=== 必要条件检查：M(s) < 0 的信号是否都不伤害？ ===")
neg = M[M.mis < 0]
print(f"M<0 的 (族,信号) 点数 = {len(neg)}，其中 harm > 0 的点数 = {(neg.harm > 0).sum()}")
print("→ 若为 0，则'M>0 是伤害的必要条件'成立")
print("\n=== M>0 但不伤害的反例 ===")
pos_notharm = M[(M.mis > 0) & (M.harm > 0)]
print(pos_notharm[["family", "signal", "mis", "auc_fail", "harm"]].round(3).to_string(index=False))
