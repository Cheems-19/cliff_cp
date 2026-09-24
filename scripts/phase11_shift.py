#!/usr/bin/env python
"""Phase 11 — P0：划分现实性 × 基线完整性。

回答两个审稿人最可能的攻击：
  (1) "随机划分下可交换性几乎平凡成立" → 加 Bemis-Murcko **scaffold 划分**，与随机并排。
  (2) "Mondrian 会伤害子群，是你选错了条件化方法" → 补两条主对手：
      **adaptive / 归一化共形**（ResidualNormalisedScore 风格）与
      **协变量偏移加权共形**（Tibshirani 2019）。

每个 (数据集, 划分方式, 种子) 跑 8 个方法，报告总体/悬崖/非悬崖覆盖与宽度，
并重算错配量所需的信号 AUC（在 scaffold 划分下是否仍然成立）。

用法：
  python scripts/phase11_shift.py --data-dir data/MoleculeACE --out results/phase11_shift \
      --seeds 5 --split both --model rf --alpha 0.10 --no-cqr
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

from rdkit import DataStructs  # noqa: E402

from cliffcp import cliffs as CL  # noqa: E402
from cliffcp import conformal as CP  # noqa: E402
from cliffcp import dispersion as DP  # noqa: E402
from cliffcp import features as F  # noqa: E402
from cliffcp.data import find_moleculeace_datasets, load_potency_csv  # noqa: E402
import phase1_signal as P1  # noqa: E402  (复用 make_model / model_uncertainty / mondrian_intervals)

SIGNALS_DUMP = ["d_wstd", "d_std", "d_iqr", "d_nb", "d_nn_sim"]


def rank_auc(score, label):
    """秩和 AUC（nan 安全）。"""
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    m = np.isfinite(score)
    score, label = score[m], label[m]
    n1 = int(label.sum())
    n0 = int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bins_by_cal(sig_cal, sig_test, k):
    """按校准侧分位数分箱（NaN 安全）。

    ⚠️ 关键坑：无训练邻居的分子在 dispersion.neighbor_label_stats 里会留下 NaN
    （d_wstd / d_std / d_iqr / d_nn_sim 的"无邻居"分支只设了 d_nb 与 d_pair_gap）。
    若把 NaN 直接交给 np.quantile，**全部边界都变成 NaN**，searchsorted 会把每个
    分子都塞进同一箱；该箱样本数必然达标，于是回退成全局分位数
    —— 整臂静默退化成 global，Δ 恰好 0.00 pp、方向一致 0/15、Wilcoxon p=nan。
    这正是首次 phase11 里 mondrian_wstd 出现"完美零效应"的原因。

    这里按主流水线的既定约定把 NaN 归入**最稀疏箱**（NaN→0，即最低信号箱）。
    """
    cal = np.asarray(sig_cal, dtype=float)
    tst = np.asarray(sig_test, dtype=float)
    cal = np.where(np.isfinite(cal), cal, 0.0)
    tst = np.where(np.isfinite(tst), tst, 0.0)
    edges = np.quantile(cal, [i / k for i in range(1, k)])
    return np.searchsorted(edges, cal), np.searchsorted(edges, tst)


def scaffold_split_seeded(smiles, seed, frac_train=0.6, frac_cal=0.2):
    """带随机性的 scaffold 划分：打乱骨架组顺序后贪心分配（组内不拆）。"""
    groups = F.scaffold_groups(smiles)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n = len(smiles)
    n_train = int(round(frac_train * n))
    n_cal = int(round(frac_cal * n))
    train, cal, test = [], [], []
    for gi in order:
        grp = groups[gi]
        if len(train) + len(grp) <= n_train:
            train.extend(grp)
        elif len(cal) + len(grp) <= n_cal:
            cal.extend(grp)
        else:
            test.extend(grp)
    return (np.asarray(sorted(train), np.int64),
            np.asarray(sorted(cal), np.int64),
            np.asarray(sorted(test), np.int64))


def knn_sim_matrix(fps_cal, fps_test):
    M = np.empty((len(fps_test), len(fps_cal)), dtype=np.float32)
    for i, fp in enumerate(fps_test):
        M[i] = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps_cal), dtype=np.float32)
    return M


def run_one_dataset(df, name, cfg, args, out_root):
    fps, keep = F.morgan_fps(df["smiles_canonical"].tolist())
    if len(fps) < args.min_size:
        print(f"  [skip] N={len(fps)} < {args.min_size}", flush=True)
        return None, None
    df = df.iloc[keep].reset_index(drop=True)
    y = df["y"].to_numpy(np.float64)
    smiles = df["smiles_canonical"].tolist()
    X = F.fps_to_matrix(fps, args.n_bits)
    n = len(fps)

    neigh_idx, neigh_sim = F.tanimoto_neighbors(
        fps, sim_threshold=cfg.sim_threshold, max_neighbors=args.max_neighbors
    )
    cl = CL.dataset_level_cliffs(neigh_idx, neigh_sim, y, cfg)
    n_cliff = int(cl["is_cliff"].sum())
    print(f"  [data] n={n} features={X.shape[1]} cliffA={n_cliff} ({n_cliff/n:.1%})", flush=True)
    if n_cliff < args.min_cliff:
        print(f"  [skip] cliff too few ({n_cliff})", flush=True)
        return None, None

    m_rows, sig_rows, dump_rows = [], [], []
    rng_master = np.random.default_rng(args.seed0)

    for split_mode in args.modes:
        for s in range(args.seeds):
            seed = int(rng_master.integers(1, 10**8))
            if split_mode == "random":
                perm = np.random.default_rng(seed).permutation(n)
                nt = int(round(args.frac_train * n)); nc = int(round(args.frac_cal * n))
                train_idx, cal_idx, test_idx = perm[:nt], perm[nt:nt + nc], perm[nt + nc:]
            else:
                train_idx, cal_idx, test_idx = scaffold_split_seeded(
                    smiles, seed, args.frac_train, args.frac_cal
                )
            if len(test_idx) < 50 or len(cal_idx) < 50:
                print(f"    [skip] split {split_mode} seed={seed} n_test={len(test_idx)} n_cal={len(cal_idx)}", flush=True)
                continue

            model = P1.make_model(args.model, seed)
            model.fit(X[train_idx].astype(np.float32), y[train_idx])
            pred = model.predict(X.astype(np.float32))

            ct = np.concatenate([cal_idx, test_idx])
            std_ct = P1.model_uncertainty(
                args.model, model, X[train_idx].astype(np.float32), y[train_idx],
                X[ct].astype(np.float32), seed, args.n_members,
            )
            rf_std_cal, rf_std_test = std_ct[:len(cal_idx)], std_ct[len(cal_idx):]

            pred_cal, y_cal = pred[cal_idx], y[cal_idx]
            pred_test, y_test = pred[test_idx], y[test_idx]

            # 机制定义（defB）：训练邻域暴露（split 相关，必须每折重算）
            expo = CL.train_neighbor_exposure(test_idx, train_idx, neigh_idx, neigh_sim, y, pred, cfg)
            cliffB = expo["exposed"][test_idx]
            cliffA = cl["is_cliff"][test_idx]          # 数据集级悬崖（划分无关）
            isc = cliffB
            # 守卫用 cliffA（划分无关、稳定）；cliffB 在大偏移下会稀疏
            if cliffA.sum() < 5 or (~cliffA).sum() < 5:
                print(f"    [skip] cliffA degenerate ({int(cliffA.sum())}) {split_mode} seed={seed}", flush=True)
                continue

            st_cal = DP.neighbor_label_stats(
                fps, y, train_idx, cal_idx, neigh_idx,
                sim_floor=args.sim_floor, k=args.knn_stats_k,
                pair_sim=cfg.sim_threshold, pair_act=cfg.act_threshold,
            )
            st_test = DP.neighbor_label_stats(
                fps, y, train_idx, test_idx, neigh_idx,
                sim_floor=args.sim_floor, k=args.knn_stats_k,
                pair_sim=cfg.sim_threshold, pair_act=cfg.act_threshold,
            )

            want = getattr(args, "method_set", None)   # None = 全部
            methods = {}
            methods["global"] = CP.global_intervals(pred_cal, y_cal, pred_test, args.alpha)

            gcal = st_cal["d_wstd"]; gtest = st_test["d_wstd"]
            lab_c, lab_t = bins_by_cal(gcal, gtest, args.n_bins)
            methods["mondrian_wstd"] = P1.mondrian_intervals(lab_c, pred_cal, y_cal, lab_t, pred_test, args.alpha, args.min_cal)

            gcal_nb = st_cal["d_nb"]; gtest_nb = st_test["d_nb"]
            lab_cn, lab_tn = bins_by_cal(gcal_nb, gtest_nb, args.n_bins)
            methods["mondrian_nb"] = P1.mondrian_intervals(lab_cn, pred_cal, y_cal, lab_tn, pred_test, args.alpha, args.min_cal)

            gcal2 = rf_std_cal; gtest2 = rf_std_test
            lab_c2, lab_t2 = bins_by_cal(gcal2, gtest2, args.n_bins)
            methods["mondrian_rfstd"] = P1.mondrian_intervals(lab_c2, pred_cal, y_cal, lab_t2, pred_test, args.alpha, args.min_cal)

            if want is None or "cluster" in want:
                methods["cluster"] = CP.cluster_intervals(
                    X[cal_idx], pred_cal, y_cal, X[test_idx], pred_test, args.alpha, seed=seed
                )

            if (not args.no_knn) and (want is None or "knn_weighted" in want):
                M = knn_sim_matrix([fps[i] for i in cal_idx], [fps[i] for i in test_idx])
                methods["knn_weighted"] = CP.knn_weighted_intervals(
                    M, pred_cal, y_cal, pred_test, args.alpha, k=args.knn_k
                )

            if want is None or "adaptive" in want:
                r_fit = CP.oob_abs_residual(model, X[train_idx], y[train_idx])
                methods["adaptive"] = CP.adaptive_intervals(
                    X[train_idx], r_fit, X[cal_idx], pred_cal, y_cal, X[test_idx], pred_test, args.alpha, seed=seed
                )

            if want is None or "weighted_shift" in want:
                methods["weighted_shift"] = CP.weighted_shift_intervals(
                    X[cal_idx], pred_cal, y_cal, X[test_idx], pred_test, args.alpha, seed=seed
                )

            if (not args.no_cqr) and (want is None or "cqr" in want):
                methods["cqr"] = CP.cqr_intervals(
                    X[train_idx], y[train_idx], X[cal_idx], y_cal, X[test_idx], args.alpha, seed=seed
                )

            if want is not None:
                methods = {k: v for k, v in methods.items() if k in want}
            for mname, iv in methods.items():
                cov = (y_test >= iv["lower"]) & (y_test <= iv["upper"])
                wid = iv["upper"] - iv["lower"]
                m_rows.append(dict(
                    dataset=name, split=split_mode, seed=seed, method=mname,
                    n_test=len(test_idx), n_cliffB=int(isc.sum()), n_cliffA=int(cliffA.sum()),
                    cov_all=float(cov.mean()),
                    cov_cliffB=float(cov[isc].mean()) if isc.sum() else np.nan,
                    cov_noncliffB=float(cov[~isc].mean()),
                    gapB_pp=float(100 * (cov[~isc].mean() - cov[isc].mean())) if isc.sum() else np.nan,
                    cov_cliffA=float(cov[cliffA].mean()),
                    cov_noncliffA=float(cov[~cliffA].mean()),
                    gapA_pp=float(100 * (cov[~cliffA].mean() - cov[cliffA].mean())),
                    width_all=float(wid.mean()),
                    width_cliffA=float(wid[cliffA].mean()),
                    width_noncliffA=float(wid[~cliffA].mean()),
                ))
                for j, gi in enumerate(test_idx):
                    dump_rows.append((name, split_mode, seed, int(gi), float(y_test[j]), float(pred_test[j]),
                                      bool(isc[j]), bool(cliffA[j]), bool(cov[j]), float(wid[j])))

            cov_g = (y_test >= methods["global"]["lower"]) & (y_test <= methods["global"]["upper"])
            for sname, sv, o in (("rf_std", rf_std_test, +1), ("d_wstd", st_test["d_wstd"], +1),
                                 ("d_nb", st_test["d_nb"], -1), ("d_nn_sim", st_test["d_nn_sim"], -1)):
                sv = np.asarray(sv, float) * o
                a_f = rank_auc(sv, ~cov_g)
                a_b = rank_auc(sv, isc)
                a_a = rank_auc(sv, cliffA)
                sig_rows.append(dict(dataset=name, split=split_mode, seed=seed, signal=sname,
                                     auc_fail=a_f, auc_cliffB=a_b, auc_cliffA=a_a,
                                     misalignB=a_f - a_b, misalignA=a_f - a_a))

            gl = m_rows[-len(methods)]
            print(f"    [{split_mode} s={seed}] test={len(test_idx)} "
                  f"cliffA={int(cliffA.sum())} cliffB={int(isc.sum())} | global_cov "
                  f"all={gl['cov_all']:.3f} cliffA={gl['cov_cliffA']:.3f} "
                  f"noncliffA={gl['cov_noncliffA']:.3f} gapA={gl['gapA_pp']:+.1f}pp", flush=True)

    os.makedirs(os.path.join(out_root, "dump"), exist_ok=True)
    if dump_rows:
        dd = pd.DataFrame(dump_rows, columns=["dataset", "split", "seed", "mol_idx", "y", "pred",
                                             "cliffB", "cliffA", "covered", "width"])
        dd.to_csv(os.path.join(out_root, "dump", f"{name}_molecules.csv.gz"), index=False, compression="gzip")
    if getattr(args, "dump_signals", False) and sig_rows:
        # 机制分析用：真实流水线内部的信号、分箱标签与 q（真值，避免外部重建分歧）
        sig_dump = []
        lab_map = {"mondrian_wstd": (st_cal["d_wstd"], st_test["d_wstd"]),
                   "mondrian_nb": (st_cal["d_nb"], st_test["d_nb"]),
                   "mondrian_rfstd": (rf_std_cal, rf_std_test)}
        expo_all = CL.train_neighbor_exposure(np.arange(n), train_idx, neigh_idx, neigh_sim, y, pred, cfg)
        for j, gi in enumerate(ct):
            is_cal = j < len(cal_idx)
            src = st_cal if is_cal else st_test
            k = j if is_cal else j - len(cal_idx)
            i = cal_idx[j] if is_cal else test_idx[k]
            row = dict(dataset=name, split=split_mode, seed=seed, role="cal" if is_cal else "test",
                       mol_idx=int(gi), y=float(y[gi]), pred=float(pred[gi]),
                       residual=float(abs(y[gi] - pred[gi])),
                       cliffA=bool(cl["is_cliff"][gi]), cliffB=bool(expo_all["exposed"][gi]))
            for sn in SIGNALS_DUMP:
                row[sn] = float(src[sn][k]) if sn != "rf_std" else (
                    float(rf_std_cal[k]) if is_cal else float(rf_std_test[k]))
            for mn, (gc_a, gt_a) in lab_map.items():
                lc, lt = bins_by_cal(gc_a, gt_a, args.n_bins)
                row["bin_" + mn] = int(lc[k] if is_cal else lt[k])
                iv = methods.get(mn)
                if iv is not None and not is_cal:
                    row["q_" + mn] = float(iv["q"][k])
                    row["q_glob"] = float(iv.get("q_global", np.nan))
            sig_dump.append(row)
        pd.DataFrame(sig_dump).to_csv(
            os.path.join(out_root, "dump", f"{name}_signals.csv.gz"), index=False, compression="gzip")
    return m_rows, sig_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results/phase11_shift")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--split", default="both", choices=["random", "scaffold", "both"])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--sim", type=float, default=0.85)
    ap.add_argument("--act", type=float, default=1.0)
    ap.add_argument("--frac-train", type=float, default=0.6)
    ap.add_argument("--frac-cal", type=float, default=0.2)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--max-neighbors", type=int, default=256)
    ap.add_argument("--sim-floor", type=float, default=0.5)
    ap.add_argument("--knn-stats-k", type=int, default=50)
    ap.add_argument("--knn-k", type=int, default=100)
    ap.add_argument("--min-cal", type=int, default=30)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--model", default="rf")
    ap.add_argument("--n-members", type=int, default=6)
    ap.add_argument("--features", default="ecfp", choices=["ecfp", "rdkit"])
    ap.add_argument("--no-cqr", action="store_true")
    ap.add_argument("--no-knn", action="store_true")
    ap.add_argument("--methods", default="all",
                    help="逗号分隔的方法白名单（如 global,mondrian_wstd,mondrian_nb,mondrian_rfstd）；"
                         "all = 全部。用于低成本补跑单个修复臂。")
    ap.add_argument("--min-size", type=int, default=800)
    ap.add_argument("--min-cliff", type=int, default=60)
    ap.add_argument("--max-datasets", type=int, default=15)
    ap.add_argument("--datasets", default="",
                    help="逗号分隔的数据集名过滤（子串匹配）；空=不筛")
    ap.add_argument("--seed0", type=int, default=20260922)
    ap.add_argument("--dump-signals", action="store_true",
                    help="额外落盘真实流水线内部的信号/分箱/q（机制分析用，避免外部重建分歧）")
    args = ap.parse_args()

    args.modes = ["random", "scaffold"] if args.split == "both" else [args.split]
    args.method_set = (None if args.methods.strip().lower() == "all"
                       else {m.strip() for m in args.methods.split(",") if m.strip()})
    cfg = CL.CliffConfig(sim_threshold=args.sim, act_threshold=args.act)
    os.makedirs(args.out, exist_ok=True)

    files = find_moleculeace_datasets(args.data_dir)
    if args.datasets:
        wanted = [w.strip() for w in args.datasets.split(",") if w.strip()]
        files = [p for p in files if any(w in os.path.basename(p) for w in wanted)]
    print(f"[phase11] {len(files)} dataset files; splits={args.modes}; seeds={args.seeds}", flush=True)

    all_m, all_s = [], []
    for p in files[: args.max_datasets]:
        name = os.path.basename(p).replace(".csv", "")
        print(f"=== {name}", flush=True)
        try:
            df = load_potency_csv(p, verbose=False)
            m, s = run_one_dataset(df, name, cfg, args, args.out)
            if m:
                all_m.extend(m); all_s.extend(s)
        except Exception:
            print(f"  [ERROR] {name}\n{traceback.format_exc()}", flush=True)

        if all_m:
            pd.DataFrame(all_m).to_csv(os.path.join(args.out, "METHODS.csv"), index=False)
            pd.DataFrame(all_s).to_csv(os.path.join(args.out, "SIGNALS.csv"), index=False)

    print(f"[phase11] DONE rows={len(all_m)}", flush=True)


if __name__ == "__main__":
    main()
