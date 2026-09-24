#!/usr/bin/env python
"""出版质量图：Fig 9 方法阶梯 + Fig 10 错配量-伤害散点（机制主图）。

Fig 9: 所有方法插值到边际覆盖 0.90 后的"悬崖 vs 非悬崖差距"。
       这是论文的 punchline：一条从 CQR(8.6pp) 到 nb(23.8pp) 的阶梯，
       排序与错配量预言一致。
Fig 10: x = 错配量（B−A），y = 条件化对悬崖组的影响。
       三个模型族 × 7 个信号，展示"伤害跟着错配量走"的单调关系。

用法: python scripts/plot_ladder_mechanism.py --out-dir figures
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_BETTER = "#2166ac"   # 蓝 = 改善悬崖组
C_WORSE = "#b2182b"    # 红 = 伤害悬崖组
C_NEUTRAL = "#8c8c8c"
C_CQR = "#1b7837"      # 绿 = CQR（另一类方法）

FAMILIES = [
    ("RF + ECFP", "results_phase2/C_misalignment.csv", "results_ablation/PHASE1_ABLATION.csv"),
    ("GBR + ECFP", "results_xa_gbr/C_misalignment.csv", "results_xa_gbr/PHASE1_ABLATION.csv"),
    ("D-MPNN + ECFP", "results_xa_gnn/C_misalignment.csv", "results_xa_gnn/PHASE1_ABLATION.csv"),
    ("RF + RDKit", "results_phase2/C_misalignment.csv", "results_xa_rdkit/PHASE1_ABLATION.csv"),
]
SIG2METHOD = {"d_wstd": "wstd", "d_std": "std", "d_iqr": "iqr", "d_nb": "nb",
              "d_nn_sim": "nnsim", "d_pair_gap": "pair", "rf_std": "rfstd"}


def ladder_figure(path):
    t = pd.read_csv("results_ecfp_abl/CQR_COVERAGE_MATCHED.csv")
    d = pd.concat([pd.read_csv(f) for f in glob.glob("results_ecfp_abl/dump/*.csv.gz")],
                  ignore_index=True)
    frac = d["cliffB"].mean()
    rows = []
    for _, r in t.iterrows():
        cb, marg = r["cov_cliffB"], r["marg_at"]
        cn = (marg - frac * cb) / (1 - frac)
        rows.append({"method": r["method"], "gap": (cn - cb) * 100,
                     "width_delta": r["width"] - float(t.loc[t.method == "global", "width"].iloc[0])})
    L = pd.DataFrame(rows).sort_values("gap")
    base = float(L.loc[L.method == "global", "gap"].iloc[0])

    labels, vals, colors = [], [], []
    for _, r in L.iterrows():
        m = r["method"]
        lab = {"cqr": "CQR (adaptive quantile regression)",
               "global": "global conformal (no conditioning)",
               "mondrian_wstd": "Mondrian ← label dispersion (wstd)",
               "mondrian_std": "Mondrian ← label std",
               "mondrian_nnsim": "Mondrian ← NN similarity",
               "mondrian_iqr": "Mondrian ← label IQR",
               "mondrian_pair": "Mondrian ← cliff pair present",
               "mondrian_pairgap": "Mondrian ← max label gap",
               "mondrian_rfstd": "Mondrian ← model-side variance",
               "mondrian_nb": "Mondrian ← # training neighbours"}[m]
        labels.append(lab)
        vals.append(r["gap"])
        if m == "cqr":
            colors.append(C_CQR)
        elif m == "global":
            colors.append("#333333")
        else:
            colors.append(C_BETTER if r["gap"] < base else C_WORSE)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    y = np.arange(len(L))
    ax.barh(y, vals, color=colors, edgecolor="white", zorder=3)
    ax.axvline(base, color="#333333", ls="--", lw=1.1, zorder=4)
    ax.text(base, len(L) - 0.25, f"  global: {base:.1f} pp", fontsize=8.5,
            color="#333333", va="center")
    for yi, v in zip(y, vals):
        ax.text(v + 0.25, yi, f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("coverage gap between non-cliff and cliff molecules (pp)\n"
                  "at marginal coverage = 0.90  (smaller = fairer intervals)")
    ax.set_title("Method ladder: how much of the cliff coverage gap each method closes\n"
                 "15 ChEMBL assays, 70,170 test molecules, RF + ECFP backbone")
    ax.grid(alpha=0.25, axis="x", zorder=0)
    ax.set_xlim(0, max(vals) * 1.16)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


def mechanism_figure(path):
    rows = []
    for fam, fmis, fabl in FAMILIES:
        mis = pd.read_csv(fmis)
        abl = pd.read_csv(fabl)
        m = dict(zip(mis["signal"], mis["gap_predict_minus_see"]))
        a = dict(zip(abl["method"].str.replace("mondrian_", ""),
                     abl["delta_mean"] * 100))
        for sig, meth in SIG2METHOD.items():
            if sig in m and meth in a:
                rows.append({"family": fam, "signal": meth,
                             "misalignment": m[sig], "harm_pp": a[meth]})
    M = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fam_style = {"RF + ECFP": ("o", "#b2182b"), "GBR + ECFP": ("s", "#2166ac"),
                 "D-MPNN + ECFP": ("^", "#1b7837"), "RF + RDKit": ("D", "#762a83")}
    for fam, (mk, c) in fam_style.items():
        sub = M[M.family == fam]
        ax.scatter(sub["misalignment"], sub["harm_pp"], marker=mk, s=46,
                   facecolor=c, edgecolor="white", linewidth=0.8,
                   label=fam, zorder=3)
    # nnsim 是已知的反例（错配量最大却不伤害），单独标记；
    # 拟合线只在"排除反例后的点"上拟合，并把这一点写清楚。
    ex = M[M.signal == "nnsim"]
    inl = M[M.signal != "nnsim"]
    ax.scatter(ex["misalignment"], ex["harm_pp"], marker="x", s=70,
               color="#000000", linewidth=1.6, label="nnsim (known counter-example)",
               zorder=4)
    z = np.polyfit(inl["misalignment"], inl["harm_pp"], 1)
    xs = np.linspace(inl["misalignment"].min() - 0.01, inl["misalignment"].max() + 0.01, 50)
    ax.plot(xs, np.polyval(z, xs), color="#555555", ls=":", lw=1.2, zorder=2)
    r_all = np.corrcoef(M["misalignment"], M["harm_pp"])[0, 1]
    r_in = np.corrcoef(inl["misalignment"], inl["harm_pp"])[0, 1]
    ax.text(0.03, 0.965,
            f"fit excl. nnsim:  Pearson r = {r_in:+.2f}  (n = {len(inl)})\n"
            f"all points:       Pearson r = {r_all:+.2f}  (n = {len(M)})\n"
            f"nnsim: largest |M| yet harmless — M is a screening\n"
            f"criterion, not a quantitative predictor",
            transform=ax.transAxes, fontsize=7.6, color="#444444", va="top")
    for sig, dx, dy in [("rf_std", 0.006, 0.5), ("nb", 0.006, -0.9)]:
        sub = M[M.signal == sig]
        for _, r in sub.iterrows():
            ax.annotate(sig, (r["misalignment"], r["harm_pp"]),
                        textcoords="offset points", xytext=(dx * 1000, dy * 2),
                        fontsize=7, color="#444444")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_xlabel("misalignment  M(s) = AUC(predicts failure) − AUC(sees cliff)")
    ax.set_ylabel("effect of conditioning on cliff-group coverage (pp)")
    ax.set_title("Harm tracks misalignment — with a known counter-example\n"
                 "4 model/representation families × 7 signals")
    ax.legend(fontsize=7.6, loc="lower left", frameon=False)
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}   (Pearson r excl. nnsim = {r_in:+.2f}, n={len(inl)})")
    return r_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    # 相对路径都以项目根为基准（脚本在 <root>/scripts/ 下），
    # 否则从别的目录调用时静默找不到数据文件。
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)
    os.makedirs(args.out_dir, exist_ok=True)
    ladder_figure(os.path.join(args.out_dir, "fig9_method_ladder.png"))
    mechanism_figure(os.path.join(args.out_dir, "fig10_misalignment_vs_harm.png"))


if __name__ == "__main__":
    main()
