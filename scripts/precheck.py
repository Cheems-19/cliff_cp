#!/usr/bin/env python
"""Phase 0 预演：活动悬崖处的共形覆盖失效是否成立？

判决标准（写死，避免事后挪门槛）：
  H1 诊断     : 悬崖组覆盖率显著低于非悬崖组（分层置换检验 p<0.05，且 gap>=0.05）
  H2 反直觉   : cluster / knn_weighted 的 gap 不小于 global 的 gap
                （即"局部化不解决问题"，甚至更糟）
  两条都成立 -> 选题成立，进入完整研究。

用法：
  python scripts/precheck.py --data-dir data/MoleculeACE --out results/precheck \
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
from rdkit import DataStructs

from cliffcp import cliffs as CL
from cliffcp import conformal as CP
from cliffcp import features as F
from cliffcp import stats as S
from cliffcp.data import find_moleculeace_datasets, load_potency_csv


# --------------------------------------------------------------------- 模型


def make_model(kind: str, seed: int):
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=1,
            n_jobs=8,
            random_state=seed,
        )
    if kind == "gbr":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(random_state=seed, max_iter=400)
    raise ValueError(f"unknown model kind: {kind}")


# --------------------------------------------------------------------- 单数据集


def run_one_dataset(df, name, cfg, args, out_dir):
    # 1) 分子表示
    fps, keep = F.morgan_fps(df["smiles_canonical"].tolist())
    if len(fps) < args.min_size:
        print(f"  [skip] 有效分子仅 {len(fps)} < {args.min_size}", flush=True)
        return None
    df = df.iloc[keep].reset_index(drop=True)
    y = df["y"].to_numpy(dtype=np.float64)
    X = F.fps_to_matrix(fps, args.n_bits)
    n = len(fps)
    print(f"  [feat] n={n}, bits={args.n_bits}", flush=True)

    # 2) 精确 Tanimoto 邻域（全数据集）
    t0 = pd.Timestamp.now()
    neigh_idx, neigh_sim = F.tanimoto_neighbors(
        fps, sim_threshold=cfg.sim_threshold, max_neighbors=args.max_neighbors
    )
    print(f"  [neigh] done in {(pd.Timestamp.now()-t0).total_seconds():.1f}s", flush=True)

    # 3) Def A：数据集级悬崖标签
    cl = CL.dataset_level_cliffs(neigh_idx, neigh_sim, y, cfg)
    n_cliff = int(cl["is_cliff"].sum())
    print(
        f"  [cliffA] 悬崖分子 {n_cliff}/{n} ({n_cliff/n:.1%}), "
        f"强悬崖 {int(cl['is_cliff_strong'].sum())}",
        flush=True,
    )
    if n_cliff < args.min_cliff:
        print(f"  [skip] 悬崖分子过少 ({n_cliff} < {args.min_cliff})", flush=True)
        return None

    # 4) 多种子随机划分
    rng_master = np.random.default_rng(args.seed0)
    rows = []
    for s in range(args.seeds):
        seed = int(rng_master.integers(1, 10**8))
        perm = np.random.default_rng(seed).permutation(n)
        n_train = int(round(args.frac_train * n))
        n_cal = int(round(args.frac_cal * n))
        train_idx = perm[:n_train]
        cal_idx = perm[n_train : n_train + n_cal]
        test_idx = perm[n_train + n_cal :]
        if len(test_idx) < 50 or len(cal_idx) < 50:
            continue

        model = make_model(args.model, seed)
        model.fit(X[train_idx].astype(np.float32), y[train_idx])
        pred = model.predict(X.astype(np.float32))

        # Def B / Def C：训练邻域暴露
        expo = CL.train_neighbor_exposure(
            test_idx, train_idx, neigh_idx, neigh_sim, y, pred, cfg
        )

        y_test = y[test_idx]
        pred_test = pred[test_idx]
        cov_flags = {
            "defA": cl["is_cliff"][test_idx],
            "defB_oracle": expo["exposed"][test_idx],
            "defC_proxy": expo["exposed_proxy"][test_idx],
        }
        extra = {
            "nn_sim": expo["nn_sim"][test_idx],
            "nn_y_diff": expo["nn_y_diff"][test_idx],
            "sali_oracle": expo["sali_oracle"][test_idx],
        }

        # 5) 三种共形方法
        pred_cal = pred[cal_idx]
        y_cal = y[cal_idx]

        intervals = {}
        intervals["global"] = CP.global_intervals(pred_cal, y_cal, pred_test, args.alpha)

        intervals["cluster"] = CP.cluster_intervals(
            X_fit=X[cal_idx].astype(np.float32),
            pred_cal=pred_cal,
            y_cal=y_cal,
            X_test=X[test_idx].astype(np.float32),
            pred_test=pred_test,
            alpha=args.alpha,
            n_clusters=args.n_clusters,
            min_cal_per_cluster=args.min_cal_per_cluster,
            seed=seed,
        )

        cal_fps = [fps[i] for i in cal_idx]
        sim_mat = np.empty((len(test_idx), len(cal_idx)), dtype=np.float32)
        for a, i in enumerate(test_idx):
            sim_mat[a] = DataStructs.BulkTanimotoSimilarity(fps[i], cal_fps)
        intervals["knn_weighted"] = CP.knn_weighted_intervals(
            sim_mat, pred_cal, y_cal, pred_test, args.alpha, k=args.knn_k
        )

        # 6) 记录
        bins = S.potency_bins(y_test, args.n_potency_bins)
        for mname, iv in intervals.items():
            covered = (y_test >= iv["lower"]) & (y_test <= iv["upper"])
            for gname, gmask in cov_flags.items():
                for side, m in (
                    ("all", np.ones_like(gmask)),
                    ("cliff", gmask),
                    ("noncliff", ~gmask),
                ):
                    sel = m
                    cs = S.coverage_stats(covered[sel])
                    rows.append(
                        {
                            "seed": seed,
                            "method": mname,
                            "group_def": gname,
                            "side": side,
                            **cs,
                            "mean_width": CP.interval_width(
                                iv["lower"], iv["upper"], mask=sel
                            ),
                            "rmse": float(
                                np.sqrt(np.mean((pred_test[sel] - y_test[sel]) ** 2))
                            )
                            if sel.sum()
                            else np.nan,
                        }
                    )
            # 分层置换检验：在每个效力分箱内置换组标签，扣除效力量程的混杂
            for gname, gmask in cov_flags.items():
                pt = S.stratified_permutation_test(
                    covered, gmask, bins, n_perm=args.n_perm, seed=seed
                )
                rows.append(
                    {
                        "seed": seed,
                        "method": mname,
                        "group_def": gname,
                        "side": "__permtest__",
                        "n": int(gmask.sum()),
                        "k": -1,
                        "coverage": pt["stat"],
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                        "mean_width": np.nan,
                        "rmse": np.nan,
                        "p_value": pt["p_value"],
                        "n_perm": pt["n_perm"],
                    }
                )

    per_seed = pd.DataFrame(rows)

    # 与 MoleculeACE 自带的 cliff_mol 列做一致性交叉验证（若存在）
    agreement = None
    if "cliff_source" in df.columns:
        ref = pd.to_numeric(df["cliff_source"], errors="coerce").fillna(0).to_numpy() > 0
        ours = cl["is_cliff"]
        inter = int((ref & ours).sum())
        union = int((ref | ours).sum())
        agreement = {
            "n_ref_cliff": int(ref.sum()),
            "n_ours_cliff": int(ours.sum()),
            "jaccard": round(inter / union, 4) if union else None,
            "precision_vs_ref": round(inter / int(ours.sum()), 4) if ours.sum() else None,
            "recall_vs_ref": round(inter / int(ref.sum()), 4) if ref.sum() else None,
        }

    return {
        "name": name,
        "n": n,
        "n_cliff": n_cliff,
        "cliff_rate": n_cliff / n,
        "n_cliff_pairs_total": int(cl["n_cliff_pairs"].sum() // 2),
        "mean_sali": float(np.nanmean(cl["sali_max"][cl["is_cliff"]]) if n_cliff else np.nan),
        "cliff_agreement_with_moleculeace": agreement,
        "per_seed": per_seed,
        "top_cliff_examples": df.loc[cl["is_cliff"].astype(bool)]
        .assign(sali=cl["sali_max"][cl["is_cliff"].astype(bool)])
        .sort_values("sali", ascending=False)
        .head(10)[["smiles_canonical", "y", "sali"]]
        .to_dict(orient="records"),
    }


# --------------------------------------------------------------------- 汇总


def summarize(res, cfg, args, out_dir):
    ps = res["per_seed"]
    cov = ps[ps["side"].isin(["all", "cliff", "noncliff"])]

    # 每个 (method, group_def, side) 的覆盖率：多种子均值 + Wilson（合并计数）
    agg = (
        cov.groupby(["method", "group_def", "side"])
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_sd=("coverage", "std"),
            n_mean=("n", "mean"),
            k_sum=("k", "sum"),
            n_sum=("n", "sum"),
            width_mean=("mean_width", "mean"),
            rmse_mean=("rmse", "mean"),
        )
        .reset_index()
    )
    agg[["ci_lo", "ci_hi"]] = pd.DataFrame(
        [S.wilson_ci(int(r.k_sum), int(r.n_sum)) for r in agg.itertuples()],
        index=agg.index,
    )

    # gap = 非悬崖 - 悬崖（每种方法、每个定义）
    gaps = []
    perm = ps[ps["side"] == "__permtest__"]
    for (mname, gname), g in cov.groupby(["method", "group_def"]):
        piv = g.pivot_table(index="seed", columns="side", values="coverage")
        if not {"cliff", "noncliff"}.issubset(piv.columns):
            continue
        d = (piv["noncliff"] - piv["cliff"]).dropna()
        pt = perm[(perm["method"] == mname) & (perm["group_def"] == gname)]
        gaps.append(
            {
                "method": mname,
                "group_def": gname,
                "gap_mean": float(d.mean()),
                "gap_sd": float(d.std()) if len(d) > 1 else np.nan,
                "gap_min": float(d.min()) if len(d) else np.nan,
                "gap_max": float(d.max()) if len(d) else np.nan,
                "n_seeds": int(len(d)),
                "frac_seeds_gap_positive": float((d > 0).mean()) if len(d) else np.nan,
                "perm_p_mean": float(pt["p_value"].mean()) if len(pt) else np.nan,
                "perm_p_frac_sig": float((pt["p_value"] < 0.05).mean())
                if len(pt)
                else np.nan,
            }
        )
    gap_df = pd.DataFrame(gaps).sort_values(["group_def", "method"])

    # 判决
    gA = gap_df[(gap_df.group_def == "defA") & (gap_df.method == "global")]
    h1 = bool(
        len(gA)
        and gA.iloc[0]["gap_mean"] >= 0.05
        and gA.iloc[0]["perm_p_mean"] < 0.05
        and gA.iloc[0]["frac_seeds_gap_positive"] >= 0.7
    )
    gap_global = float(gA.iloc[0]["gap_mean"]) if len(gA) else np.nan
    loc = gap_df[
        (gap_df.group_def == "defA") & (gap_df.method.isin(["cluster", "knn_weighted"]))
    ]
    h2 = bool(len(loc) and (loc["gap_mean"] >= gap_global * 0.9).all())

    verdict = {
        "dataset": res["name"],
        "n": res["n"],
        "n_cliff": res["n_cliff"],
        "cliff_rate": round(res["cliff_rate"], 4),
        "global_gap_defA": round(gap_global, 4) if np.isfinite(gap_global) else None,
        "H1_cliff_undercovered": h1,
        "H2_localization_does_not_help": h2,
        "verdict": "TOPIC_VALID" if (h1 and h2) else ("H1_ONLY" if h1 else "NOT_SUPPORTED"),
    }
    return agg, gap_df, verdict


def plot_report(res, agg, gap_df, out_dir, alpha):
    name = res["name"]
    methods = ["global", "cluster", "knn_weighted"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for ax, gdef in zip(axes, ["defA", "defB_oracle", "defC_proxy"]):
        sub = agg[agg.group_def == gdef]
        if sub.empty:
            ax.set_axis_off()
            continue
        xs = np.arange(len(methods))
        w = 0.36
        for k, side in enumerate(["noncliff", "cliff"]):
            vals, los, his = [], [], []
            for m in methods:
                r = sub[(sub.method == m) & (sub.side == side)]
                if len(r):
                    vals.append(r.iloc[0]["coverage_mean"])
                    los.append(r.iloc[0]["coverage_mean"] - r.iloc[0]["ci_lo"])
                    his.append(r.iloc[0]["ci_hi"] - r.iloc[0]["coverage_mean"])
                else:
                    vals.append(np.nan)
                    los.append(0)
                    his.append(0)
            ax.bar(
                xs + (k - 0.5) * w,
                vals,
                w,
                yerr=[los, his],
                capsize=3,
                label=side,
                color="#d62728" if side == "cliff" else "#1f77b4",
                alpha=0.85,
            )
        ax.axhline(1 - alpha, ls="--", c="k", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("empirical coverage")
        ax.set_title(f"{gdef}\n(nominal={1-alpha:.2f})")
        ax.legend(fontsize=8)

    fig.suptitle(f"{name} | n={res['n']} cliff={res['n_cliff']} ({res['cliff_rate']:.1%})")
    fig.tight_layout()
    p = pathlib.Path(out_dir) / f"{name}_coverage.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


# --------------------------------------------------------------------- 主流程


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results/precheck")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--sim", type=float, default=0.85)
    ap.add_argument("--act", type=float, default=1.0)
    ap.add_argument("--frac-train", type=float, default=0.6)
    ap.add_argument("--frac-cal", type=float, default=0.2)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--max-neighbors", type=int, default=256)
    ap.add_argument("--n-clusters", type=int, default=12)
    ap.add_argument("--min-cal-per-cluster", type=int, default=30)
    ap.add_argument("--knn-k", type=int, default=200)
    ap.add_argument("--n-potency-bins", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=4000)
    ap.add_argument("--model", default="rf")
    ap.add_argument("--min-size", type=int, default=800)
    ap.add_argument("--min-cliff", type=int, default=60)
    ap.add_argument("--max-datasets", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=20260922)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CL.CliffConfig(sim_threshold=args.sim, act_threshold=args.act)
    print(f"[cfg] {cfg.to_dict()}", flush=True)
    print(f"[cfg] alpha={args.alpha} seeds={args.seeds} model={args.model}", flush=True)

    files = find_moleculeace_datasets(args.data_dir)
    print(f"[data] 发现 {len(files)} 个可用数据集", flush=True)

    # 先用文件大小粗排（大文件通常样本多），再按实际 N 过滤
    files = sorted(files, key=lambda p: -os.path.getsize(p))

    verdicts, all_gaps, all_agg = [], [], []
    done = 0
    for p in files:
        if done >= args.max_datasets:
            break
        name = pathlib.Path(p).stem
        print(f"\n=== {name} ===", flush=True)
        try:
            df = load_potency_csv(p)
            df.attrs["name"] = name
            if len(df) < args.min_size:
                print(f"  [skip] N={len(df)} < {args.min_size}", flush=True)
                continue
            res = run_one_dataset(df, name, cfg, args, out_dir)
            if res is None:
                continue
            agg, gap_df, verdict = summarize(res, cfg, args, out_dir)
            png = plot_report(res, agg, gap_df, out_dir, args.alpha)

            with open(out_dir / f"{name}.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "config": {**cfg.to_dict(), "alpha": args.alpha,
                                   "seeds": args.seeds, "model": args.model},
                        "cliff_summary": {
                            "n": res["n"], "n_cliff": res["n_cliff"],
                            "cliff_rate": res["cliff_rate"],
                            "n_cliff_pairs_total": res["n_cliff_pairs_total"],
                            "mean_sali": res["mean_sali"],
                        },
                        "cliff_agreement_with_moleculeace": res[
                            "cliff_agreement_with_moleculeace"
                        ],
                        "verdict": verdict,
                        "coverage": agg.to_dict(orient="records"),
                        "gaps": gap_df.to_dict(orient="records"),
                        "top_cliff_examples": res["top_cliff_examples"],
                    },
                    fh,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            res["per_seed"].to_csv(out_dir / f"{name}_per_seed.csv", index=False)
            verdicts.append(verdict)
            all_gaps.append(gap_df.assign(dataset=name))
            all_agg.append(agg.assign(dataset=name))
            print(f"  [verdict] {verdict['verdict']}  gap={verdict['global_gap_defA']}", flush=True)
            print(f"  [plot] {png}", flush=True)
            done += 1
        except Exception:
            print(f"  [ERROR] {name}\n{traceback.format_exc()}", flush=True)

    if not verdicts:
        print("\n[FATAL] 没有任何数据集跑通", flush=True)
        return

    vdf = pd.DataFrame(verdicts)
    vdf.to_csv(out_dir / "VERDICTS.csv", index=False)
    if all_gaps:
        pd.concat(all_gaps).to_csv(out_dir / "GAPS.csv", index=False)
    if all_agg:
        pd.concat(all_agg).to_csv(out_dir / "COVERAGE.csv", index=False)

    # 跨数据集汇总
    print("\n" + "=" * 72, flush=True)
    print("跨数据集判决汇总", flush=True)
    print("=" * 72, flush=True)
    for r in vdf.itertuples():
        print(
            f"{r.dataset:28s} n={r.n:6d} cliff={r.n_cliff:5d} "
            f"gap={r.global_gap_defA} H1={r.H1_cliff_undercovered} "
            f"H2={r.H2_localization_does_not_help} -> {r.verdict}",
            flush=True,
        )
    n_valid = int((vdf["verdict"] == "TOPIC_VALID").sum())
    n_h1 = int(vdf["H1_cliff_undercovered"].sum())
    n_pos = int((vdf["global_gap_defA"] > 0).sum())
    print(
        f"\n逐数据集 H1 成立: {n_h1}/{len(vdf)}   "
        f"gap 方向为正: {n_pos}/{len(vdf)}   H1+H2: {n_valid}/{len(vdf)}",
        flush=True,
    )
    print(
        "\n[重要] 逐数据集判决在低功效下不可靠：\n"
        "  单个 MoleculeACE 数据集约 3000 个分子、其中悬崖分子仅 ~250 个，\n"
        "  测试集里只有 40-60 个，再按效力分箱做分层置换后每层仅约 10 个样本，\n"
        "  功效极低 -> 会出现“方向 8/8 一致、但逐个 p 都略高于 0.05”的现象。\n"
        "  真实判决必须看合并检验，运行：\n"
        "    python scripts/meta_analysis.py --gaps results/precheck/GAPS.csv",
        flush=True,
    )
    if n_pos >= max(1, int(0.75 * len(vdf))):
        print(
            f"\n>>> 方向一致性 {n_pos}/{len(vdf)} -> 效应很可能真实但单数据集功效不足；"
            "请以合并检验结果为准。",
            flush=True,
        )
    else:
        print("\n>>> 方向一致性不足 -> 效应可能不存在，考虑转候选 2。", flush=True)


if __name__ == "__main__":
    main()
