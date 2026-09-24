#!/usr/bin/env python
"""Phase 13 — clean 机制检验：固定预测器，只变"信号精度"。

背景（这是本文机制主张最强的检验，也是 §3.6 自己承认做不到的那个）：
此前"成员数 3→8"的实验同时改了**模型**与**信号**，无法单独归因。
这里把**预测器固定**（= 200 棵树的森林均值，全程不变），
只把"模型侧不确定性信号"的精度沿阶梯下调：用 k ∈ {2,5,10,25,50,100,200} 棵树的
子集标准差作为 σ_k。预测不变、分区与残差不变，**唯一变量是信号的精度**。

机制预言：σ_k 的精度随 k 下降 → 它对"预测失效"的判别力 AUC_fail 下降 →
错配量 M_k 下降 → 条件化对悬崖组的伤害下降（趋近 0）。
若该预言兑现，"伤害需要信号本身携带关于失效的信息"就从相关性升级为**受控实验**。

用法：
  python scripts/phase13_signal_ladder.py --data-dir data/MoleculeACE --out results/phase13 \
      --datasets A,B,C --seeds 5
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cliffcp import cliffs as CL  # noqa: E402
from cliffcp import conformal as CP  # noqa: E402
from cliffcp import dispersion as DP  # noqa: E402
from cliffcp import features as F  # noqa: E402
from cliffcp.data import find_moleculeace_datasets, load_potency_csv  # noqa: E402
import phase1_signal as P1  # noqa: E402

KS = [2, 5, 10, 25, 50, 100, 200]


def rank_auc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, bool)
    m = np.isfinite(score); score, label = score[m], label[m]
    n1 = int(label.sum()); n0 = int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def run_one_dataset(df, name, cfg, args, out_root):
    fps, keep = F.morgan_fps(df["smiles_canonical"].tolist())
    if len(fps) < args.min_size:
        return []
    df = df.iloc[keep].reset_index(drop=True)
    y = df["y"].to_numpy(np.float64)
    X = F.fps_to_matrix(fps, args.n_bits)
    n = len(fps)
    neigh_idx, neigh_sim = F.tanimoto_neighbors(fps, cfg.sim_threshold, args.max_neighbors)
    cl = CL.dataset_level_cliffs(neigh_idx, neigh_sim, y, cfg)
    if int(cl["is_cliff"].sum()) < args.min_cliff:
        print(f"  [skip] cliff too few", flush=True)
        return []
    print(f"  [data] n={n} cliffA={int(cl['is_cliff'].sum())}", flush=True)

    rows = []
    rng_master = np.random.default_rng(args.seed0)
    for s in range(args.seeds):
        seed = int(rng_master.integers(1, 10**8))
        perm = np.random.default_rng(seed).permutation(n)
        nt = int(round(args.frac_train * n)); nc = int(round(args.frac_cal * n))
        train_idx, cal_idx, test_idx = perm[:nt], perm[nt:nt + nc], perm[nt + nc:]
        if len(test_idx) < 50 or len(cal_idx) < 50:
            continue

        model = P1.make_model("rf", seed)
        model.fit(X[train_idx].astype(np.float32), y[train_idx])
        estimators = list(model.estimators_)
        n_tree = len(estimators)

        # 固定预测器：森林均值，全程不变（cal/test 与邻域暴露都用同一数组）
        Xf = X.astype(np.float32)
        pred_full = model.predict(Xf)
        ct = np.concatenate([cal_idx, test_idx])
        T = np.stack([t.predict(Xf[ct]) for t in estimators], axis=1)   # (n_ct, n_tree)
        pred_cal = pred_full[cal_idx]; pred_test = pred_full[test_idx]
        y_cal, y_test = y[cal_idx], y[test_idx]

        expo = CL.train_neighbor_exposure(test_idx, train_idx, neigh_idx, neigh_sim, y, pred_full, cfg)
        cliffB = expo["exposed"][test_idx]
        cliffA = cl["is_cliff"][test_idx]

        st_test = DP.neighbor_label_stats(
            fps, y, train_idx, test_idx, neigh_idx,
            sim_floor=args.sim_floor, k=args.knn_stats_k,
            pair_sim=cfg.sim_threshold, pair_act=cfg.act_threshold,
        )

        g = CP.global_intervals(pred_cal, y_cal, pred_test, args.alpha)
        cov_g = (y_test >= g["lower"]) & (y_test <= g["upper"])

        rng = np.random.default_rng(seed + 7)
        for k in KS:
            if k > n_tree:
                continue
            pick = rng.choice(n_tree, size=k, replace=False)
            sig_ct = T[:, pick].std(axis=1)
            sig_cal = sig_ct[:len(cal_idx)]; sig_test = sig_ct[len(cal_idx):]

            edges = np.quantile(sig_cal, [i / args.n_bins for i in range(1, args.n_bins)])
            lab_c = np.searchsorted(edges, sig_cal); lab_t = np.searchsorted(edges, sig_test)
            iv = P1.mondrian_intervals(lab_c, pred_cal, y_cal, lab_t, pred_test, args.alpha, args.min_cal)
            cov = (y_test >= iv["lower"]) & (y_test <= iv["upper"])

            a_f = rank_auc(sig_test, ~cov_g)
            a_b = rank_auc(sig_test, cliffB)
            rows.append(dict(
                dataset=name, seed=seed, k=k,
                auc_fail=a_f, auc_cliffB=a_b, misalign=a_f - a_b,
                d_cov_cliffB_pp=100 * (cov[cliffB].mean() - cov_g[cliffB].mean()) if cliffB.sum() else np.nan,
                d_cov_cliffA_pp=100 * (cov[cliffA].mean() - cov_g[cliffA].mean()),
                cov_cliffB=float(cov[cliffB].mean()) if cliffB.sum() else np.nan,
                cov_cliffA=float(cov[cliffA].mean()),
                n_cliffB=int(cliffB.sum()), n_cliffA=int(cliffA.sum()),
            ))
        print(f"    [s={seed}] test={len(test_idx)} cliffB={int(cliffB.sum())} done", flush=True)
    return rows


def summarize(all_rows, out_root):
    R = pd.DataFrame(all_rows)
    R.to_csv(os.path.join(out_root, "LADDER.csv"), index=False)
    lines = []

    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

    p("=" * 70)
    p("Phase 13：固定预测器、只变信号精度（RF 树子集 std）")
    p("=" * 70)
    p("rows:", len(R), "| datasets:", R.dataset.nunique(), "| seeds:", R.seed.nunique())
    p("")
    p(f"{'k':>5s} {'AUC_fail':>9s} {'AUC_cliffB':>11s} {'misalign':>9s} "
      f"{'dCovB(pp)':>10s} {'dCovA(pp)':>10s} {'harm<0':>7s}")
    for k in sorted(R.k.unique()):
        g = R[R.k == k]
        # 逐数据集先跨种子平均
        d = g.groupby("dataset").agg(mis=("misalign", "mean"), dcB=("d_cov_cliffB_pp", "mean"),
                                     dcA=("d_cov_cliffA_pp", "mean"), af=("auc_fail", "mean"),
                                     ab=("auc_cliffB", "mean")).dropna()
        p(f"{k:5d} {d.af.mean():9.3f} {d.ab.mean():11.3f} {d.mis.mean():+9.3f} "
          f"{d.dcB.mean():+10.2f} {d.dcA.mean():+10.2f} {int((d.dcA < 0).sum()):>3d}/{len(d)}")
    p("")
    p("判读：若 AUC_fail 与 |dCov| 随 k 单调上升 → 机制预言兑现（信号越有信息，伤害越大）。")
    with open(os.path.join(out_root, "SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results/phase13")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--sim", type=float, default=0.85)
    ap.add_argument("--act", type=float, default=1.0)
    ap.add_argument("--frac-train", type=float, default=0.6)
    ap.add_argument("--frac-cal", type=float, default=0.2)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--max-neighbors", type=int, default=256)
    ap.add_argument("--sim-floor", type=float, default=0.5)
    ap.add_argument("--knn-stats-k", type=int, default=50)
    ap.add_argument("--min-cal", type=int, default=30)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--min-size", type=int, default=800)
    ap.add_argument("--min-cliff", type=int, default=60)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--max-datasets", type=int, default=15)
    ap.add_argument("--seed0", type=int, default=20260922)
    args = ap.parse_args()

    cfg = CL.CliffConfig(sim_threshold=args.sim, act_threshold=args.act)
    os.makedirs(args.out, exist_ok=True)
    files = find_moleculeace_datasets(args.data_dir)
    if args.datasets:
        wanted = [w.strip() for w in args.datasets.split(",") if w.strip()]
        files = [p for p in files if any(w in os.path.basename(p) for w in wanted)]
    print(f"[phase13] {len(files)} datasets; seeds={args.seeds}", flush=True)

    all_rows = []
    for p in files[: args.max_datasets]:
        name = os.path.basename(p).replace(".csv", "")
        print(f"=== {name}", flush=True)
        try:
            df = load_potency_csv(p, verbose=False)
            all_rows.extend(run_one_dataset(df, name, cfg, args, args.out))
        except Exception:
            print(f"  [ERROR] {name}\n{traceback.format_exc()}", flush=True)
        if all_rows:
            pd.DataFrame(all_rows).to_csv(os.path.join(args.out, "LADDER.csv"), index=False)
    if all_rows:
        summarize(all_rows, args.out)
    print(f"[phase13] DONE rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
