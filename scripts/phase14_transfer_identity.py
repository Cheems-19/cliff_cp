# -*- coding: utf-8 -*-
"""Phase 14 — 覆盖转移恒等式：为"错配信号为什么会伤害悬崖组"给出可证明的分解。

## 理论（先写在这里，再在数据上验证）

设测试分子 i 的非一致性分数 eps_i = |y_i - pred_i|，全局分位数 q̄（由校准集得到），
按信号 s 分箱后该分子所在箱的分位数为 q_{k(i)}。则对任意子群 G：

    ΔCov_G := Cov_G(Mondrian_s) - Cov_G(global)
            = (1/|G|) * [ #{i∈G: q̄ < eps_i <= q_{k(i)}}     (新被覆盖：加宽救回)
                        - #{i∈G: q_{k(i)} < eps_i <= q̄} ]   (新被遗漏：收紧致失)

这是**恒等式**（对给定的 eps 与 q 逐项成立，无分布假设）。它把"伤害"精确分解为
"G 有多少失效落进了被收紧的箱"减去"G 有多少失效落在被加宽的箱"。

**推论（伤害判据）**：s 伤害 G  ⟺  G 的失效更多落在被收紧的箱里。
而"收紧"发生在校准分数小的箱（信号判定为"容易"的区域）——于是：
若某个信号把 G 所在的区域判为"容易"（低分箱）却并不预测失效，G 的失效就会落进收紧箱 → 伤害。
这正是错配量 M(s)=AUC(失效)-AUC(悬崖) > 0 的**机制解释**。

本脚本在既有分子级 dump 上：
  (a) 验证恒等式（应逐分子精确成立）
  (b) 检验"收紧质量"是否预测伤害的符号与幅度（跨 8 信号 × 多族）
输出：results_phase14/TRANSFER_*.csv + SUMMARY.txt
"""
import glob
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

SIGNALS = {
    "d_wstd": ("covered_wstd", "q_wstd"),
    "d_std": ("covered_std", "q_std"),
    "d_iqr": ("covered_iqr", "q_iqr"),
    "d_nb": ("covered_nb", "q_nb"),
    "d_nn_sim": ("covered_nnsim", "q_nnsim"),
    "d_pair_gap": ("covered_pairgap", "q_pairgap"),
    "d_pair": ("covered_pair", "q_pair"),
    "rf_std": ("covered_rfstd", "q_rfstd"),
}
RUNS = {
    "rf_ecfp": "results_ecfp_abl/dump",
    "rf_rdkit": "results_xa_rdkit/dump",
}

os.makedirs("results_phase14", exist_ok=True)


def load_dump_dir(d):
    frames = []
    for p in sorted(glob.glob(os.path.join(d, "*.csv.gz"))):
        g = pd.read_csv(p)
        g["dataset"] = os.path.basename(p).replace("_molecules.csv.gz", "")
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def main():
    lines = []

    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

    p("=" * 78)
    p("Phase 14 — 覆盖转移恒等式：验证 + 伤害判据检验")
    p("=" * 78)

    id_rows, harm_rows = [], []
    for run, d in RUNS.items():
        if not os.path.isdir(d):
            continue
        df = load_dump_dir(d)
        qg = df["q_global"].to_numpy(float)
        eps = np.abs(df["y"].to_numpy(float) - df["pred"].to_numpy(float))
        cliff = df["cliffB"].to_numpy(bool)
        p(f"\n### run={run}  molecules={len(df)}  datasets={df.dataset.nunique()}  cliffB={cliff.mean():.4f}")

        for sname, (cov_col, q_col) in SIGNALS.items():
            qs = df[q_col].to_numpy(float)
            ok = np.isfinite(qs) & np.isfinite(qg)
            for ds, idx in df.groupby("dataset").groups.items():
                idx = np.asarray(idx)
                idx = idx[ok[idx]]
                if len(idx) < 50:
                    continue
                e, qq, q0 = eps[idx], qs[idx], qg[idx]
                cm = cliff[idx]
                if cm.sum() < 5:
                    continue
                lo = np.minimum(qq, q0); hi = np.maximum(qq, q0)
                recovered = (e > lo) & (e <= hi) & (qq > q0)     # 加宽救回
                missed = (e > lo) & (e <= hi) & (qq < q0)        # 收紧致失
                # 恒等式两侧
                lhs = (np.mean(e <= qq) - np.mean(e <= q0))
                rhs = (recovered.sum() - missed.sum()) / len(idx)
                id_rows.append(dict(run=run, dataset=ds, signal=sname,
                                    lhs=lhs, rhs=rhs, resid=abs(lhs - rhs)))
                # 伤害判据：收紧质量 vs 观测伤害（悬崖组）
                nG = int(cm.sum())
                frac_tight = float((qq[cm] < q0[cm] - 1e-12).mean())
                frac_wide = float((qq[cm] > q0[cm] + 1e-12).mean())
                d_cov_cliff = float(np.mean(e[cm] <= qq[cm]) - np.mean(e[cm] <= q0[cm]))
                # 幅度加权（相对收紧/加宽）—— 捕捉"收紧幅度大但只影响少数箱"的情形
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = (q0[cm] - qq[cm]) / np.maximum(q0[cm], 1e-12)
                rel = rel[np.isfinite(rel)]
                tight_mag = float(np.mean(np.clip(rel, 0, None))) if rel.size else 0.0
                wide_mag = float(np.mean(np.clip(-rel, 0, None))) if rel.size else 0.0
                harm_rows.append(dict(run=run, dataset=ds, signal=sname,
                                      frac_tight=frac_tight, frac_wide=frac_wide,
                                      net_tight=frac_tight - frac_wide,
                                      tight_mag=tight_mag, wide_mag=wide_mag,
                                      net_mag=wide_mag - tight_mag,
                                      d_cov_cliff=d_cov_cliff,
                                      tightened_miss_frac=float(missed[cm].mean()),
                                      recovered_frac=float(recovered[cm].mean())))

    I = pd.DataFrame(id_rows); H = pd.DataFrame(harm_rows)
    I.to_csv("results_phase14/TRANSFER_IDENTITY.csv", index=False)
    H.to_csv("results_phase14/TRANSFER_HARM.csv", index=False)

    p("\n### (a) 恒等式验证（应 ~1e-15）")
    p(f"  n={len(I)}  最大残差={I.resid.max():.3e}  中位残差={I.resid.median():.3e}")

    p("\n### (b) 伤害判据：净收紧质量 vs 观测 ΔCov（悬崖组）")
    p("  注：ΔCov<0 = 伤害悬崖组；net_tight>0 = 悬崖组被更多地收紧")
    for run in H.run.unique():
        for sname in SIGNALS:
            g = H[(H.run == run) & (H.signal == sname)]
            if len(g) == 0:
                continue
            p(f"  [{run}] {sname:11s} net_tight={g.net_tight.mean():+.3f}  "
              f"ΔCov_cliff={g.d_cov_cliff.mean()*100:+6.2f}pp  "
              f"({int((g.d_cov_cliff < 0).sum())}/{len(g)} 数据集受害)")

    p("\n### (c) 相关性（逐 (run, signal) 聚合到数据集级）")
    agg = H.groupby(["run", "signal"]).mean(numeric_only=True).reset_index()
    for run in agg.run.unique():
        a = agg[agg.run == run]
        r1 = np.corrcoef(a.net_tight, a.d_cov_cliff)[0, 1]
        r2 = np.corrcoef(a.tightened_miss_frac, a.d_cov_cliff)[0, 1]
        r3 = np.corrcoef(a.net_mag, a.d_cov_cliff)[0, 1]
        p(f"  [{run}] corr(net_tight,ΔCov)={r1:+.3f} | corr(收紧致失占比,ΔCov)={r2:+.3f} | "
          f"corr(net_mag,ΔCov)={r3:+.3f}  (n={len(a)})")
    a = agg
    p(f"  合并: corr(net_tight,ΔCov)={np.corrcoef(a.net_tight, a.d_cov_cliff)[0,1]:+.3f} | "
      f"corr(net_mag,ΔCov)={np.corrcoef(a.net_mag, a.d_cov_cliff)[0,1]:+.3f}  (n={len(a)})")
    p("")
    p("### (d) 可部署性：net_mag / net_tight 只需校准集 + 信号 + 悬崖掩码（无测试标签）")
    p("  → 可作为部署前诊断：先算目标组的净收紧质量，再决定是否用该信号做条件化。")

    with open("results_phase14/SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[saved] results_phase14/SUMMARY.txt", flush=True)


if __name__ == "__main__":
    main()
