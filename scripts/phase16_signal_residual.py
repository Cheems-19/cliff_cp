#!/usr/bin/env python
"""Phase 16 —— C3：为什么 d_nn_sim 错配量最大却不伤害悬崖组（净加宽）？

对 random 划分的每个 (数据集, 种子)：
  - 拟合 RF，预测**全体**分子
  - 计算**全体**分子的信号（d_wstd / d_std / d_iqr / d_nb / d_nn_sim / rf_std）
  - 记录残差 |y − pred|、cliffA / cliffB、角色（cal / test）

机制假设（本脚本用于检验）：
  Mondrian 条件化下悬崖组被加宽还是收紧，取决于**悬崖分子与"大残差分子"是否同箱**——
  d_nn_sim 把悬崖与高相似的大残差分子分到同一箱（箱内残差尾部重 → q 大 → 加宽）；
  d_nb 把悬崖与稠密易预测分子分到同一箱（箱内残差尾部轻 → q 小 → 收紧）。

输出：results/phase16_signal_residual/dump/{dataset}.csv.gz
用法：
  python scripts/phase16_signal_residual.py --data-dir data/MoleculeACE \
      --out results/phase16_signal_residual --seeds 3
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
from cliffcp import dispersion as DP  # noqa: E402
from cliffcp import features as F  # noqa: E402
from cliffcp.data import find_moleculeace_datasets, load_potency_csv  # noqa: E402
import phase1_signal as P1  # noqa: E402

SIGNALS = ["d_wstd", "d_std", "d_iqr", "d_nb", "d_nn_sim"]


def run_one_dataset(df, name, cfg, args):
    fps, keep = F.morgan_fps(df["smiles_canonical"].tolist())
    if len(fps) < args.min_size:
        print(f"  [skip] N={len(fps)}", flush=True)
        return []
    df = df.iloc[keep].reset_index(drop=True)
    y = df["y"].to_numpy(np.float64)
    X = F.fps_to_matrix(fps, args.n_bits)
    n = len(fps)

    neigh_idx, neigh_sim = F.tanimoto_neighbors(
        fps, sim_threshold=cfg.sim_threshold, max_neighbors=args.max_neighbors)
    cl = CL.dataset_level_cliffs(neigh_idx, neigh_sim, y, cfg)
    cliffA_all = cl["is_cliff"]
    n_cliff = int(cliffA_all.sum())
    print(f"  [data] n={n} cliffA={n_cliff}", flush=True)
    if n_cliff < args.min_cliff:
        print("  [skip] cliff too few", flush=True)
        return []

    rows = []
    rng_master = np.random.default_rng(args.seed0)
    for s in range(args.seeds):
        seed = int(rng_master.integers(1, 10 ** 8))
        perm = np.random.default_rng(seed).permutation(n)
        nt = int(round(args.frac_train * n)); nc = int(round(args.frac_cal * n))
        train_idx, cal_idx, test_idx = perm[:nt], perm[nt:nt + nc], perm[nt + nc:]

        model = P1.make_model(args.model, seed)
        model.fit(X[train_idx].astype(np.float32), y[train_idx])
        pred = model.predict(X.astype(np.float32))
        residual = np.abs(y - pred)

        ct = np.concatenate([cal_idx, test_idx])
        std_ct = P1.model_uncertainty(
            args.model, model, X[train_idx].astype(np.float32), y[train_idx],
            X[ct].astype(np.float32), seed, args.n_members)
        rf_std = np.full(n, np.nan)
        rf_std[ct] = std_ct

        st = {}
        for role, idxs in (("cal", cal_idx), ("test", test_idx)):
            st[role] = DP.neighbor_label_stats(
                fps, y, train_idx, idxs, neigh_idx,
                sim_floor=args.sim_floor, k=args.knn_stats_k,
                pair_sim=cfg.sim_threshold, pair_act=cfg.act_threshold)

        role_of = np.full(n, "", dtype=object)
        role_of[cal_idx] = "cal"
        role_of[test_idx] = "test"

        expo = CL.train_neighbor_exposure(np.arange(n), train_idx, neigh_idx, neigh_sim, y, pred, cfg)
        cliffB_all = expo["exposed"]

        pos_cal = {int(i): k for k, i in enumerate(cal_idx)}
        pos_test = {int(i): k for k, i in enumerate(test_idx)}
        n_cal = len(cal_idx)
        for i in ct:
            is_cal = role_of[i] == "cal"
            src = st["cal"] if is_cal else st["test"]
            k = pos_cal[i] if is_cal else pos_test[i]
            k_std = k if is_cal else n_cal + k          # std_ct 按 cal 在前拼接
            row = dict(dataset=name, seed=seed, role=role_of[i], mol_idx=int(i),
                       y=float(y[i]), pred=float(pred[i]), residual=float(residual[i]),
                       cliffA=bool(cliffA_all[i]), cliffB=bool(cliffB_all[i]),
                       rf_std=float(std_ct[k_std]) if np.isfinite(std_ct[k_std]) else np.nan)
            for sn in SIGNALS:
                row[sn] = float(src[sn][k])
            rows.append(row)

        nb = int((cliffB_all[ct]).sum())
        print(f"    [s={seed}] cal={len(cal_idx)} test={len(test_idx)} cliffB(ct)={nb}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results/phase16_signal_residual")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sim", type=float, default=0.85)
    ap.add_argument("--act", type=float, default=1.0)
    ap.add_argument("--frac-train", type=float, default=0.6)
    ap.add_argument("--frac-cal", type=float, default=0.2)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--max-neighbors", type=int, default=256)
    ap.add_argument("--sim-floor", type=float, default=0.5)
    ap.add_argument("--knn-stats-k", type=int, default=50)
    ap.add_argument("--model", default="rf")
    ap.add_argument("--n-members", type=int, default=6)
    ap.add_argument("--min-size", type=int, default=800)
    ap.add_argument("--min-cliff", type=int, default=60)
    ap.add_argument("--max-datasets", type=int, default=15)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--seed0", type=int, default=20260923)
    args = ap.parse_args()

    cfg = CL.CliffConfig(sim_threshold=args.sim, act_threshold=args.act)
    os.makedirs(args.out, exist_ok=True)
    files = find_moleculeace_datasets(args.data_dir)
    if args.datasets:
        wanted = [w.strip() for w in args.datasets.split(",") if w.strip()]
        files = [p for p in files if any(w in os.path.basename(p) for w in wanted)]
    print(f"[phase16] {len(files)} datasets; seeds={args.seeds}", flush=True)

    all_rows = []
    for p in files[: args.max_datasets]:
        name = os.path.basename(p).replace(".csv", "")
        print(f"=== {name}", flush=True)
        try:
            df = load_potency_csv(p, verbose=False)
            rows = run_one_dataset(df, name, cfg, args)
            if rows:
                all_rows.extend(rows)
                dd = pd.DataFrame(rows)
                os.makedirs(os.path.join(args.out, "dump"), exist_ok=True)
                dd.to_csv(os.path.join(args.out, "dump", f"{name}_signals.csv.gz"),
                          index=False, compression="gzip")
        except Exception:
            print(f"  [ERROR] {name}\n{traceback.format_exc()}", flush=True)

    print(f"[phase16] DONE rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
