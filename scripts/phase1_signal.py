#!/usr/bin/env python
"""Phase 1：无标签预警信号的评估，以及基于它的条件化校准。

三个问题一次回答：

  Q1  哪个无标签信号能检出"会落在区间外"的分子？
      —— 用 AUC（阈值无关），并带两个参照：
         naive_proxy = |预测 - 最近训练邻居标签|（上一轮失败的信号）
         oracle      = |真实标签 - 最近训练邻居标签|（用了测试标签，ACU 上界）

  Q2  把信号当 Mondrian 分区做条件化校准，能不能把最差分组覆盖率拉回名义值？
      —— 对比 global CP：边缘覆盖率必须不掉、最差分组覆盖率要升、宽度代价要有界

  Q3  代价是多少？—— 平均区间宽度、以及高风险组的宽度

用法：
  python scripts/phase1_signal.py --data-dir data/MoleculeACE --out results/phase1 \
      --seeds 10 --alpha 0.1 --sim 0.85 --act 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from cliffcp import cliffs as CL
from cliffcp import conformal as CP
from cliffcp import dispersion as DP
from cliffcp import features as F
from cliffcp import stats as S
from cliffcp.data import find_moleculeace_datasets, load_potency_csv

# 信号的天然方向：+1 表示"越大越不可靠"，-1 表示"越大越可靠"
DET_ORIENT = {
    "d_wstd": +1,
    "d_std": +1,
    "d_range": +1,
    "d_iqr": +1,
    "d_pair_gap": +1,
    "d_nb": -1,
    "d_nn_sim": -1,
    "rf_std": +1,
    "naive_proxy": +1,
    "oracle": +1,
}


def make_model(kind: str, seed: int, n_jobs: int = 8):
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=200, min_samples_leaf=1, n_jobs=n_jobs, random_state=seed
        )
    if kind == "gbr":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            random_state=seed, max_iter=300, learning_rate=0.1, max_features=0.4
        )
    raise ValueError(f"unknown model kind: {kind}")


def model_uncertainty(kind, model, X_tr, y_tr, X_target, seed, n_members=6):
    """模型内部不确定性 —— 论文里那个"内敌"信号。

    跨架构必须换算法，但要归到同一个名字（内部 tag 仍叫 rfstd），
    否则分析脚本无法跨架构复用：
      RF  -> 树间预测标准差（架构自带，200 棵树天然是一组集成）
      GBR -> K 个不同随机种子的成员模型预测标准差（集成方差，GBR 的对应物）

    注意 GBR 的随机性来源：HistGradientBoosting 若不做特征子采样，
    给定数据几乎是确定性的，多种子集成的方差会趋近 0（假信号）。
    因此上面 make_model 里对 GBR 设了 max_features=0.4 以制造真实的多样性。
    """
    if kind == "rf" and getattr(model, "estimators_", None):
        preds = np.stack([t.predict(X_target) for t in model.estimators_], axis=1)
        return preds.std(axis=1)

    if kind == "gbr":
        from sklearn.ensemble import HistGradientBoostingRegressor

        preds = []
        for k in range(n_members):
            m = HistGradientBoostingRegressor(
                random_state=seed + 1000 * (k + 1),
                max_iter=300,
                learning_rate=0.1,
                max_features=0.4,
            )
            m.fit(X_tr, y_tr)
            preds.append(m.predict(X_target))
        return np.stack(preds, axis=1).std(axis=1)

    return np.full(len(X_target), np.nan)


def mondrian_intervals(
    groups_cal, pred_cal, y_cal, groups_test, pred_test, alpha, min_cal=30
):
    """Mondrian 条件化校准：按组取分位数，组内样本不足则回退全局分位数。"""
    scores = np.abs(np.asarray(y_cal, float) - np.asarray(pred_cal, float))
    q_global = CP.conformal_quantile(scores, alpha)
    groups_cal = np.asarray(groups_cal)
    groups_test = np.asarray(groups_test)

    q_map, n_fallback = {}, 0
    for g in np.unique(groups_cal):
        m = groups_cal == g
        if int(m.sum()) >= min_cal:
            q_map[g] = CP.conformal_quantile(scores[m], alpha)
        else:
            q_map[g] = q_global
            n_fallback += 1

    q_test = np.array([q_map.get(g, q_global) for g in groups_test], dtype=float)
    p = np.asarray(pred_test, float)
    return {
        "q": q_test,
        "lower": p - q_test,
        "upper": p + q_test,
        "q_global": q_global,
        "n_fallback": n_fallback,
    }


def run_one_dataset(df, name, cfg, args, out_dir):
    fps, keep = F.morgan_fps(df["smiles_canonical"].tolist())
    if len(fps) < args.min_size:
        print(f"  [skip] N={len(fps)} < {args.min_size}", flush=True)
        return None
    df = df.iloc[keep].reset_index(drop=True)
    y = df["y"].to_numpy(dtype=np.float64)
    # chemprop 走分子图输入，需要保留 SMILES（顺序与过滤后的 df 一致）
    smiles = df["smiles_canonical"].tolist()

    # 预测器看到的特征（可切换表示）；悬崖判定与邻域检索一律用 ECFP 指纹，
    # 这样"表示"是唯一被改变的变量。
    if args.features == "rdkit":
        X, _desc_names = F.rdkit_desc_matrix(df["smiles_canonical"].tolist())
    else:
        X = F.fps_to_matrix(fps, args.n_bits)
    n = len(fps)
    print(f"  [feat] n={n}, features={args.features}, dim={X.shape[1]}", flush=True)

    neigh_idx, neigh_sim = F.tanimoto_neighbors(
        fps, sim_threshold=cfg.sim_threshold, max_neighbors=args.max_neighbors
    )
    cl = CL.dataset_level_cliffs(neigh_idx, neigh_sim, y, cfg)
    n_cliff = int(cl["is_cliff"].sum())
    print(f"  [cliffA] {n_cliff}/{n} ({n_cliff/n:.1%})", flush=True)
    if n_cliff < args.min_cliff:
        print(f"  [skip] cliff too few ({n_cliff})", flush=True)
        return None

    rng_master = np.random.default_rng(args.seed0)
    det_rows, bin_rows, m_rows, sig_rows, dump_rows = [], [], [], [], []

    for s in range(args.seeds):
        seed = int(rng_master.integers(1, 10**8))
        perm = np.random.default_rng(seed).permutation(n)
        n_train = int(round(args.frac_train * n))
        n_cal = int(round(args.frac_cal * n))
        train_idx, cal_idx, test_idx = (
            perm[:n_train],
            perm[n_train : n_train + n_cal],
            perm[n_train + n_cal :],
        )
        if len(test_idx) < 50 or len(cal_idx) < 50:
            continue

        if args.model == "chemprop":
            # 第三个模型族：D-MPNN（分子图消息传递网络）。
            # 只有"预测器看到的输入"换成分子图；悬崖判定与邻域检索仍用 ECFP，
            # 使"模型族"成为唯一变量（与跨架构验证的设计一致）。
            # 这里对全体分子预测（GPU 上便宜），保持 pred 是全量数组，
            # 下游的 train_neighbor_exposure / 分箱逻辑一行都不用改。
            from cliffcp import gnn

            pred, std_all = gnn.train_ensemble_and_predict(
                smiles, y, train_idx, np.arange(n),
                n_members=args.n_members, epochs=args.gnn_epochs,
                seed=seed, accelerator=args.accelerator, gpu=args.gpu,
                workdir=args.gnn_workdir,
            )
            rf_std_cal = std_all[cal_idx]
            rf_std_test = std_all[test_idx]
        else:
            model = make_model(args.model, seed)
            model.fit(X[train_idx].astype(np.float32), y[train_idx])
            pred = model.predict(X.astype(np.float32))

            # 模型自身的无标签不确定性：作为"内敌"信号。
            # 跨架构时用不同算法实现（RF 树间方差 / GBR 多种子集成方差），
            # 但对外统一叫 rfstd，以便 phase2/phase3/phase1_analyze 三个脚本跨架构复用。
            ct_idx = np.concatenate([cal_idx, test_idx])
            Xct = X[ct_idx].astype(np.float32)
            std_ct = model_uncertainty(
                args.model, model,
                X[train_idx].astype(np.float32), y[train_idx],
                Xct, seed, args.n_members,
            )
            rf_std_cal = std_ct[: len(cal_idx)]
            rf_std_test = std_ct[len(cal_idx) :]

        pred_cal, y_cal, pred_test, y_test = (
            pred[cal_idx],
            y[cal_idx],
            pred[test_idx],
            y[test_idx],
        )

        # --- 全局 CP 基线
        g_iv = CP.global_intervals(pred_cal, y_cal, pred_test, args.alpha)
        g_cov = (y_test >= g_iv["lower"]) & (y_test <= g_iv["upper"])

        # --- 无标签信号（校准集与测试集分别算，都用训练集标签）
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

        # --- 参照信号：naive proxy（上一轮失败的）与 oracle（用测试标签的上界）
        # 注意：neighbor_label_stats 返回的数组已按 target_idx 顺序对齐，
        # 不要再拿全局索引去索引一遍（早期版本在这里越界，且越界才暴露）。
        assert len(st_test["d_nb"]) == len(test_idx), "signal arrays must align with test_idx"
        expo = CL.train_neighbor_exposure(
            test_idx, train_idx, neigh_idx, neigh_sim, y, pred, cfg
        )
        signals = dict(st_test)
        signals["naive_proxy"] = expo["nn_pred_diff"][test_idx]
        signals["oracle"] = expo["nn_y_diff"][test_idx]
        signals["rf_std"] = rf_std_test

        # --- Q1：AUC
        bad = ~g_cov
        for dname, sig in signals.items():
            orient = DET_ORIENT.get(dname, +1)
            auc = S.auc_detect(orient * np.asarray(sig, float), bad)
            det_rows.append(
                {
                    "seed": seed,
                    "detector": dname,
                    "auc": auc,
                    "n_bad": int(bad.sum()),
                    "n_test": int(bad.size),
                }
            )

        # --- Q2/Q3：方法对比
        # 分区变量消融：把每个候选信号都做成一个 Mondrian 分区。
        # 这一步是用来诚实回答"对齐是不是这一族离散度量的共同性质"——
        # 如果 d_std / d_iqr 也都能改善悬崖组，那 d_wstd 就不是独有优势，
        # 论文里必须把主张降级为"这一族都可用，d_wstd 只是其中最好"。
        methods = {
            "global": g_iv,
            "mondrian_pair": mondrian_intervals(
                groups_cal=(st_cal["d_pair_gap"] >= cfg.act_threshold).astype(int),
                pred_cal=pred_cal, y_cal=y_cal,
                groups_test=(st_test["d_pair_gap"] >= cfg.act_threshold).astype(int),
                pred_test=pred_test, alpha=args.alpha, min_cal=args.min_cal,
            ),
        }

        # 外部强对手：共形分位数回归（CQR）。区间不对称、随 x 自适应，
        # 是"只赢绝对残差基线不算赢"的那条线，必须正面比。
        # 但它每个划分要额外拟合两个 HGB 分位数模型，在 α 扫描这类
        # "同一模型只换覆盖率"的批跑里是纯浪费——实测把单数据集耗时从
        # ~1.3 min 拉到 ~6.7 min（约 5×），因此可用 --no-cqr 跳过；
        # CQR 只需在主配置（α=0.10）里跑一次作为对手。
        if not args.no_cqr:
            methods["cqr"] = CP.cqr_intervals(
                X[train_idx], y[train_idx],
                X[cal_idx], y[cal_idx],
                X[test_idx], args.alpha, seed=seed,
            )

        partition_sources = {
            "wstd": (st_cal["d_wstd"], st_test["d_wstd"]),
            "std": (st_cal["d_std"], st_test["d_std"]),
            "iqr": (st_cal["d_iqr"], st_test["d_iqr"]),
            "nb": (st_cal["d_nb"], st_test["d_nb"]),
            "nnsim": (st_cal["d_nn_sim"], st_test["d_nn_sim"]),
            "pairgap": (st_cal["d_pair_gap"], st_test["d_pair_gap"]),
            "rfstd": (rf_std_cal, rf_std_test),
        }
        # 分箱边界取自"校准集侧的信号分布"，与测试标签无关，分区合法性更干净
        part_edges = {}
        for tag, (sig_cal, sig_test) in partition_sources.items():
            e = S.bin_edges_from(sig_cal, args.n_bins)
            part_edges[tag] = e
            methods[f"mondrian_{tag}"] = mondrian_intervals(
                groups_cal=S.assign_bins(sig_cal, e),
                pred_cal=pred_cal, y_cal=y_cal,
                groups_test=S.assign_bins(sig_test, e),
                pred_test=pred_test, alpha=args.alpha, min_cal=args.min_cal,
            )

        edges = part_edges["wstd"]
        edges_rf = part_edges["rfstd"]

        # 评估分组：oracle 悬崖暴露组 + 信号五分位组
        eval_groups = {
            "cliffB_oracle": expo["exposed"][test_idx],
            "cliffA": cl["is_cliff"][test_idx],
        }
        bin_specs = [
            ("worst_d_wstd", st_test["d_wstd"], S.bin_edges_from(st_test["d_wstd"], args.n_bins)),
            ("worst_rf_std", rf_std_test, S.bin_edges_from(rf_std_test, args.n_bins)),
        ]

        # 分子级落盘：把信号值、覆盖状态、各分区的分位数都存下来。
        # 用途：Phase 2 要直接测量"分区错配"——即
        #   ① 某信号能否看见悬崖（AUC: 信号 -> 是否 cliffB）
        #   ② 同一信号能否预测失效（AUC: 信号 -> 是否被欠覆盖）
        # 只有落到分子级才能做交叉表与分组分解，否则只能停在推理。
        if args.dump_dir:
            rec = {
                "dataset": name,
                "seed": seed,
                "mol_idx": test_idx,
                "y": y_test,
                "pred": pred_test,
                "rf_std": rf_std_test,
                "d_wstd": np.asarray(st_test["d_wstd"], float),
                "d_std": np.asarray(st_test["d_std"], float),
                "d_iqr": np.asarray(st_test["d_iqr"], float),
                "d_nb": np.asarray(st_test["d_nb"], float),
                "d_nn_sim": np.asarray(st_test["d_nn_sim"], float),
                "d_pair_gap": np.asarray(st_test["d_pair_gap"], float),
                "cliffA": cl["is_cliff"][test_idx],
                "cliffB": expo["exposed"][test_idx],
                "nn_y_diff": expo["nn_y_diff"][test_idx],
                "covered_global": g_cov,
                "q_global": g_iv["q"],
                "bin_rfstd": S.assign_bins(rf_std_test, edges_rf),
                "bin_wstd": S.assign_bins(st_test["d_wstd"], edges),
            }
            for tag, iv in methods.items():
                if tag == "global":
                    continue
                rec[f"covered_{tag.replace('mondrian_', '')}"] = (
                    (y_test >= iv["lower"]) & (y_test <= iv["upper"])
                )
                rec[f"q_{tag.replace('mondrian_', '')}"] = iv["q"]
            dump_rows.append(pd.DataFrame(rec))

        # 记录各组的关键信号均值。
        # 用途：验证本文的核心机理解释 —— 模型自身不确定性（rf_std）在悬崖处是否偏低。
        # 如果 rf_std(cliff) < rf_std(noncliff)，就说明"模型不确定性看不见悬崖"，
        # 于是把 rf_std 当校准分区会把悬崖分子分进高置信组、发更窄的区间。
        for gname, gmask in eval_groups.items():
            for side, m in (("cliff", gmask), ("noncliff", ~gmask)):
                if int(np.sum(m)) == 0:
                    continue
                sig_rows.append(
                    {
                        "dataset": name, "seed": seed,
                        "group_def": gname, "side": side,
                        "n": int(np.sum(m)),
                        "rf_std_mean": float(np.nanmean(rf_std_test[m])),
                        "d_wstd_mean": float(np.nanmean(np.asarray(st_test["d_wstd"], float)[m])),
                        "d_std_mean": float(np.nanmean(np.asarray(st_test["d_std"], float)[m])),
                        "d_nb_mean": float(np.nanmean(np.asarray(st_test["d_nb"], float)[m])),
                        "d_nn_sim_mean": float(np.nanmean(np.asarray(st_test["d_nn_sim"], float)[m])),
                        "d_pair_gap_mean": float(np.nanmean(np.asarray(st_test["d_pair_gap"], float)[m])),
                    }
                )

        for mname, iv in methods.items():
            cov = (y_test >= iv["lower"]) & (y_test <= iv["upper"])
            for gname, gmask in eval_groups.items():
                for side, m in (
                    ("all", np.ones_like(gmask)),
                    ("cliff", gmask),
                    ("noncliff", ~gmask),
                ):
                    cs = S.coverage_stats(cov[m])
                    m_rows.append(
                        {
                            "seed": seed, "method": mname, "group_def": gname,
                            "side": side, **cs,
                            "mean_width": CP.interval_width(iv["lower"], iv["upper"], mask=m),
                        }
                    )

            # 各分箱变量下的最差组覆盖率与离散度
            row = {
                "seed": seed, "method": mname, "group_def": "__bins__",
                "side": "__summary__", "n": -1, "k": -1,
                "coverage": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "mean_width": CP.interval_width(iv["lower"], iv["upper"]),
            }
            for tag, sig, eg in bin_specs:
                rows_g, summ = S.coverage_by_group(sig, cov, eg)
                row[f"{tag}_cov"] = summ.get("worst_group_coverage", np.nan)
                row[f"{tag}_spread"] = summ.get("coverage_spread", np.nan)
                for b in rows_g:
                    bin_rows.append(
                        {"seed": seed, "method": mname, "bin_var": tag, **b}
                    )
            m_rows.append(row)

    det = pd.DataFrame(det_rows)
    mtd = pd.DataFrame(m_rows)
    bins = pd.DataFrame(bin_rows)
    sig = pd.DataFrame(sig_rows)

    if args.dump_dir and dump_rows:
        dd = pathlib.Path(args.dump_dir)
        dd.mkdir(parents=True, exist_ok=True)
        pd.concat(dump_rows, ignore_index=True).to_csv(
            dd / f"{name}_molecules.csv.gz", index=False, compression="gzip"
        )
        print(f"  [dump] {len(dump_rows)} 个划分的分子级记录已落盘", flush=True)

    return {
        "name": name, "n": n, "n_cliff": n_cliff,
        "detectors": det, "methods": mtd, "bins": bins, "signals": sig,
    }


def summarize(res, args, out_dir):
    det = res["detectors"]
    dsum = (
        det.groupby("detector")
        .agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"), n_seeds=("auc", "size"))
        .reset_index()
        .sort_values("auc_mean", ascending=False)
    )

    mtd = res["methods"]
    cov = mtd[mtd["side"].isin(["all", "cliff", "noncliff"])]
    msum = (
        cov.groupby(["method", "group_def", "side"])
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_sd=("coverage", "std"),
            k_sum=("k", "sum"), n_sum=("n", "sum"),
            width_mean=("mean_width", "mean"),
        )
        .reset_index()
    )
    msum[["ci_lo", "ci_hi"]] = pd.DataFrame(
        [S.wilson_ci(int(r.k_sum), int(r.n_sum)) for r in msum.itertuples()],
        index=msum.index,
    )

    bsum = (
        mtd[mtd["side"] == "__summary__"]
        .groupby("method")
        .agg(
            worst_d_wstd=("worst_d_wstd_cov", "mean"),
            spread_d_wstd=("worst_d_wstd_spread", "mean"),
            worst_rf_std=("worst_rf_std_cov", "mean"),
            spread_rf_std=("worst_rf_std_spread", "mean"),
            width_mean=("mean_width", "mean"),
        )
        .reset_index()
    )

    # 边缘覆盖率（判断条件化是否牺牲了有效性）
    # 注意：side=="all" 的行与 group_def 无关（同一批测试点的总覆盖率），
    # 直接 set_index 会得到重复索引、merge 后行数翻倍（早期版本就踩了这个坑）。
    marg = (
        msum[msum.side == "all"]
        .groupby("method")[["coverage_mean", "ci_lo", "ci_hi", "width_mean"]]
        .mean()
        .rename(columns=lambda c: "marg_" + c)
    )
    bsum = bsum.merge(marg, left_on="method", right_index=True, how="left")

    return dsum, msum, bsum


def plot_report(res, dsum, bsum, out_dir, alpha):
    name = res["name"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    ax = axes[0]
    d = dsum.dropna(subset=["auc_mean"]).sort_values("auc_mean")
    ax.barh(d["detector"], d["auc_mean"], xerr=d["auc_sd"], capsize=3,
            color="#1f77b4", alpha=0.85)
    ax.axvline(0.5, ls="--", c="k", lw=1)
    ax.set_xlabel("AUC (predicting miscoverage)")
    ax.set_title("Q1 detector quality\n(0.5 = no skill)")

    ax = axes[1]
    bins_w = res["bins"]
    if "bin_var" in bins_w.columns:
        bins_w = bins_w[bins_w["bin_var"] == "worst_d_wstd"]
    b = bins_w.groupby(["method", "bin"]).agg(
        coverage=("coverage", "mean"), n=("n", "mean")
    ).reset_index()
    for m, sub in b.groupby("method"):
        ax.plot(sub["bin"], sub["coverage"], "o-", label=m)
    ax.axhline(1 - alpha, ls="--", c="k", lw=1, label="nominal")
    ax.set_xlabel("local label dispersion (d_wstd) quintile")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Q2 coverage vs local dispersion")
    ax.legend(fontsize=8)

    fig.suptitle(f"{name} | n={res['n']} cliff={res['n_cliff']}")
    fig.tight_layout()
    p = pathlib.Path(out_dir) / f"{name}_phase1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results/phase1")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--sim", type=float, default=0.85)
    ap.add_argument("--act", type=float, default=1.0)
    ap.add_argument("--frac-train", type=float, default=0.6)
    ap.add_argument("--frac-cal", type=float, default=0.2)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--max-neighbors", type=int, default=256)
    ap.add_argument("--sim-floor", type=float, default=0.5,
                    help="判定为训练邻居的相似度下限")
    ap.add_argument("--knn-stats-k", type=int, default=50,
                    help="计算局部标签统计量时取多少个近邻")
    ap.add_argument("--min-cal", type=int, default=30)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--model", default="rf")
    ap.add_argument("--no-cqr", action="store_true",
                    help="跳过 CQR 外部对手。CQR 每个划分要额外拟合两个 HGB 分位数模型，"
                         "在 α 扫描这类'同一模型换覆盖率'的批跑里是纯粹的浪费"
                         "（实测把单数据集耗时从 ~1.3 min 拉到 ~6.7 min）。"
                         "CQR 只需要在主配置（α=0.10）里跑一次作对手。")
    ap.add_argument("--features", default="ecfp", choices=["ecfp", "rdkit"],
                    help="预测器使用的分子表示；悬崖判定与邻域检索始终用 ECFP")
    ap.add_argument("--n-members", type=int, default=6,
                    help="GBR 集成成员数（RF 不需要，用树间方差）；"
                         "chemprop 下即 D-MPNN 集成成员数")
    ap.add_argument("--gnn-epochs", type=int, default=30,
                    help="chemprop 的训练轮数（必须大于其 warmup=2）")
    ap.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"],
                    help="chemprop 的训练设备。CPU 实测 34–127 s/epoch，"
                         "30 epoch 需 15–65 min/模型，规模化只能在 GPU 上跑")
    ap.add_argument("--gpu", default="0", help="GPU 序号（经 CUDA_VISIBLE_DEVICES 传入）")
    ap.add_argument("--gnn-workdir", default="results/gnn_work",
                    help="chemprop 中间产物目录（CSV/checkpoint/预测）")
    ap.add_argument("--dump-dir", default="",
                    help="非空则把分子级记录写到这里（供 Phase 2 错配分析）")
    ap.add_argument("--min-size", type=int, default=800)
    ap.add_argument("--min-cliff", type=int, default=60)
    ap.add_argument("--max-datasets", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=20260922)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = CL.CliffConfig(sim_threshold=args.sim, act_threshold=args.act)
    print(f"[cfg] {cfg.to_dict()} alpha={args.alpha} seeds={args.seeds}", flush=True)

    files = sorted(find_moleculeace_datasets(args.data_dir),
                   key=lambda p: -os.path.getsize(p))
    all_det, all_mtd, all_bin = [], [], []
    all_sig = []
    done = 0
    for p in files:
        if done >= args.max_datasets:
            break
        name = pathlib.Path(p).stem
        print(f"\n=== {name} ===", flush=True)
        try:
            df = load_potency_csv(p)
            if len(df) < args.min_size:
                print(f"  [skip] N={len(df)}", flush=True)
                continue
            res = run_one_dataset(df, name, cfg, args, out_dir)
            if res is None:
                continue
            dsum, msum, bsum = summarize(res, args, out_dir)
            png = plot_report(res, dsum, bsum, out_dir, args.alpha)

            with open(out_dir / f"{name}.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "config": {**cfg.to_dict(), "alpha": args.alpha,
                                   "seeds": args.seeds, "model": args.model,
                                   "sim_floor": args.sim_floor, "k": args.knn_stats_k},
                        "dataset": {"name": name, "n": res["n"], "n_cliff": res["n_cliff"]},
                        "detectors": dsum.to_dict(orient="records"),
                        "methods": bsum.to_dict(orient="records"),
                        "coverage_table": msum.to_dict(orient="records"),
                        "plot": png,
                    }, fh, ensure_ascii=False, indent=2, default=str,
                )
            res["detectors"].to_csv(out_dir / f"{name}_detectors.csv", index=False)
            res["methods"].to_csv(out_dir / f"{name}_methods.csv", index=False)
            res["bins"].to_csv(out_dir / f"{name}_bins.csv", index=False)
            res["signals"].to_csv(out_dir / f"{name}_signals.csv", index=False)

            all_det.append(res["detectors"].assign(dataset=name))
            all_mtd.append(res["methods"].assign(dataset=name))
            all_bin.append(res["bins"].assign(dataset=name))
            all_sig.append(res["signals"].assign(dataset=name))

            print("  [detectors]", flush=True)
            for r in dsum.itertuples():
                print(f"    {r.detector:14s} AUC={r.auc_mean:.4f} +- {r.auc_sd:.4f}", flush=True)
            print("  [methods vs signal quintile]", flush=True)
            for r in bsum.itertuples():
                print(
                    f"    {r.method:16s} marg={r.marg_coverage_mean:.4f} "
                    f"worst(d_wstd)={r.worst_d_wstd:.4f} spread={r.spread_d_wstd:.4f} "
                    f"worst(rf_std)={r.worst_rf_std:.4f} width={r.width_mean:.3f}",
                    flush=True,
                )
            done += 1
        except Exception:
            print(f"  [ERROR] {name}\n{traceback.format_exc()}", flush=True)

    if not all_det:
        print("\n[FATAL] 没有数据集跑通", flush=True)
        return

    D = pd.concat(all_det)
    M = pd.concat(all_mtd)
    B = pd.concat(all_bin)
    Sg = pd.concat(all_sig)
    D.to_csv(out_dir / "DETECTORS.csv", index=False)
    M.to_csv(out_dir / "METHODS.csv", index=False)
    B.to_csv(out_dir / "BINS.csv", index=False)
    Sg.to_csv(out_dir / "SIGNALS_BY_GROUP.csv", index=False)

    print("\n" + "=" * 78, flush=True)
    print("机理解释验证：模型自身不确定性(rf_std)在悬崖组是否偏低", flush=True)
    print("=" * 78, flush=True)
    for gd in Sg["group_def"].unique():
        sub = Sg[Sg["group_def"] == gd]
        piv = sub.groupby(["dataset", "seed", "side"])["rf_std_mean"].mean().unstack()
        if {"cliff", "noncliff"}.issubset(piv.columns):
            diff = (piv["cliff"] - piv["noncliff"]).dropna()
            if len(diff):
                try:
                    pv = stats.wilcoxon(diff).pvalue if not np.allclose(diff, 0) else np.nan
                except Exception:
                    pv = np.nan
                print(
                    f"  {gd:14s} rf_std(cliff)-rf_std(noncliff) = "
                    f"{diff.mean():+.4f}  (n={len(diff)}, wilcoxon p={pv:.2e})",
                    flush=True,
                )
    print(
        "\n  负值 = 悬崖组被模型认为更有把握 -> 拿到更窄区间 -> 解释为何按 rf_std "
        "分区会恶化悬崖组",
        flush=True,
    )

    print("\n" + "=" * 78, flush=True)
    print("跨数据集汇总：信号判别力（AUC，10 个划分 × N 个数据集的均值）", flush=True)
    print("=" * 78, flush=True)
    pool = (
        D.groupby("detector")
        .agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
             n_rows=("auc", "size"))
        .reset_index()
        .sort_values("auc_mean", ascending=False)
    )
    for r in pool.itertuples():
        print(f"  {r.detector:14s} AUC = {r.auc_mean:.4f} +- {r.auc_sd:.4f}  (n={r.n_rows})", flush=True)

    print("\n" + "=" * 78, flush=True)
    print("跨数据集汇总：条件化校准的效果", flush=True)
    print("=" * 78, flush=True)
    ms = (
        M[M["side"] == "all"]
        .groupby("method")
        .agg(marg_cov=("coverage", "mean"), width=("mean_width", "mean"))
    )
    ws = (
        M[M["side"] == "__summary__"]
        .groupby("method")
        .agg(
            worst_d_wstd=("worst_d_wstd_cov", "mean"),
            spread_d_wstd=("worst_d_wstd_spread", "mean"),
            worst_rf_std=("worst_rf_std_cov", "mean"),
            spread_rf_std=("worst_rf_std_spread", "mean"),
        )
    )
    cliffB = (
        M[(M["group_def"] == "cliffB_oracle") & (M["side"] == "cliff")]
        .groupby("method").agg(cliffB_cov=("coverage", "mean"))
    )
    tbl = ms.join(ws).join(cliffB)
    print(tbl.to_string())
    print(
        "\n解读要点：\n"
        "  marg_cov        应≈名义覆盖率（条件化不能牺牲有效性）\n"
        "  worst_d_wstd    按标签离散度五分位分组后的最差组覆盖率，越大越好\n"
        "  worst_rf_std    按模型树间方差五分位分组后的最差组覆盖率\n"
        "  cliffB_cov      机制定义的悬崖组覆盖率，越大越好\n"
        "  width           平均区间宽度，是代价",
        flush=True,
    )
    tbl.to_csv(out_dir / "SUMMARY_METHODS.csv")


if __name__ == "__main__":
    main()
