#!/usr/bin/env python
"""覆盖率匹配比较：把"匹配宽度"的诊断性归一化，升级为合法流程。

原理
----
rule_frontier.py 里的"匹配宽度"是把 q 乘一个常数——这破坏了共形的有限样本保证，
只能算诊断。严格的做法是**调 α**：每个 α 都对应一个真实可用的方法，
随着 α 变小，边缘覆盖率与宽度同时上升，于是每个方法都有一条
    {α → (边缘覆盖率, 宽度, 悬崖组覆盖率)}
的轨迹。把这条轨迹插值到"边缘覆盖率 = 0.90"处，读出的悬崖组覆盖率
就是**同等有效性下的**结果，无需任何 hack。

用法:
    python scripts/coverage_matched.py --root results --glob "alpha_*" [--target 0.90]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

# 目录名里的 α 标签：alpha_007 / cqr_007 都要能匹配。
# 第一版写死 alpha_(\d+)$，导致 cqr_* 的四个目录全部被跳过、
# 报"没找到 METHODS.csv"（目录明明存在）——查了半天 glob，结果是正则的问题。
TAG_RE = re.compile(r"^[a-z]+_(\d+)$")


def load(root, pattern):
    frames = []
    for d in sorted(glob.glob(os.path.join(root, pattern))):
        f = os.path.join(d, "METHODS.csv")
        if not os.path.isfile(f):
            continue
        m = TAG_RE.search(os.path.basename(d).rstrip("/"))
        if not m:
            continue
        alpha = int(m.group(1)) / 100.0
        t = pd.read_csv(f)
        t["alpha"] = alpha
        t["run"] = os.path.basename(d)
        frames.append(t)
    if not frames:
        raise SystemExit(f"没找到 {root}/{pattern}/METHODS.csv")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--glob", default="alpha_*")
    ap.add_argument("--target", type=float, default=0.90,
                    help="要对齐到的边缘覆盖率（默认 0.90）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    M = load(args.root, args.glob)
    print(f"载入 {len(M):,} 行 | α 取值 "
          f"{sorted(M['alpha'].unique().tolist())} | 方法 {M['method'].nunique()} 个\n")

    # 逐 (run, dataset, seed, method) 先算好两个视角的量，再对 (dataset, seed) 取均值
    marg = M[M.side == "all"].copy()
    cliffB = M[(M.group_def == "cliffB_oracle") & (M.side == "cliff")].copy()
    cliffA = M[(M.group_def == "cliffA") & (M.side == "cliff")].copy()

    keys = ["run", "alpha", "method", "dataset", "seed"]
    # 坑：METHODS.csv 里 side=="all" 的行会按 group_def（cliffA / cliffB_oracle）各出现一次，
    # 直接 merge 会把每个 (dataset,seed) 重复一份（实测 n_pairs 变成 120 而不是 60）。
    # 均值不受影响，但样本数和任何离散度指标都会翻倍 —— 必须先去重。
    marg = marg[keys + ["coverage", "mean_width"]].drop_duplicates(
        subset=["run", "alpha", "method", "dataset", "seed"]
    ).rename(columns={"coverage": "marg_cov", "mean_width": "width"})
    cliffB = cliffB[keys + ["coverage"]].rename(columns={"coverage": "cov_cliffB"})
    cliffA = cliffA[keys + ["coverage"]].rename(columns={"coverage": "cov_cliffA"})
    T = marg.merge(cliffB, on=keys, how="left").merge(cliffA, on=keys, how="left")

    # 先按 (run, alpha, method) 汇总
    S = T.groupby(["run", "alpha", "method"]).agg(
        marg_cov=("marg_cov", "mean"),
        width=("width", "mean"),
        cov_cliffB=("cov_cliffB", "mean"),
        cov_cliffA=("cov_cliffA", "mean"),
        n_pairs=("seed", "size"),
    ).reset_index()
    print("=" * 104)
    print("原始 α 轨迹（每个 α 就是一个真实可用的方法）")
    print("=" * 104)
    print(S.sort_values(["method", "alpha"]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------- 插值到目标边缘覆盖率 ----------------
    print("\n" + "=" * 104)
    print(f"插值到边缘覆盖率 = {args.target:.2f} 处（同等有效性下的悬崖组覆盖率）")
    print("=" * 104)
    rows = []
    for m, sub in S.groupby("method"):
        sub = sub.sort_values("marg_cov")
        x = sub["marg_cov"].to_numpy(float)
        if len(x) < 2 or x.min() > args.target or x.max() < args.target:
            rows.append({"method": m, "note": "α 范围未覆盖目标覆盖率",
                         "marg_at": np.nan, "cov_cliffB": np.nan,
                         "cov_cliffA": np.nan, "width": np.nan})
            continue
        cb = np.interp(args.target, x, sub["cov_cliffB"].to_numpy(float))
        ca = np.interp(args.target, x, sub["cov_cliffA"].to_numpy(float))
        wd = np.interp(args.target, x, sub["width"].to_numpy(float))
        # 目标覆盖率对应的 α（线性插值）
        al = np.interp(args.target, x, sub["alpha"].to_numpy(float))
        rows.append({"method": m, "note": "", "alpha_for_target": al,
                     "marg_at": args.target,
                     "cov_cliffB": cb, "cov_cliffA": ca, "width": wd})
    I = pd.DataFrame(rows).sort_values("cov_cliffB", ascending=False)
    print(I.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    base_row = I[I.method == "global"]
    if not base_row.empty and np.isfinite(base_row.iloc[0]["cov_cliffB"]):
        base_cb = float(base_row.iloc[0]["cov_cliffB"])
        base_w = float(base_row.iloc[0]["width"])
        print(f"\n以 global 为基准（悬崖组 {base_cb:.4f}，宽度 {base_w:.4f}）：")
        for r in I.itertuples():
            if r.method == "global" or not np.isfinite(r.cov_cliffB):
                continue
            print(f"  {r.method:16s} 悬崖组 Δ = {(r.cov_cliffB - base_cb)*100:+.2f} pp"
                  f" | 宽度 Δ = {r.width - base_w:+.4f}"
                  f" ({(r.width/base_w - 1)*100:+.1f}%)")

    # ---------------- 逐配对插值 + 配对检验 ----------------
    # 上面那张表是"池化均值再插值"，没有显著性。
    # 严格做法：对每个 (dataset, seed) 各自插值到边缘覆盖 = target，
    # 得到该配对上的悬崖组覆盖率，再做与 global 的配对 Wilcoxon；
    # 并统计逐数据集的方向一致性（防 Simpson 悖论）。
    print("\n" + "=" * 104)
    print(f"逐配对插值（每个 (dataset, seed) 各自插到边缘覆盖 {args.target:.2f}）+ 配对 Wilcoxon")
    print("=" * 104)
    from scipy import stats as _st

    methods = sorted(T["method"].unique())
    if "global" not in methods:
        raise SystemExit("没有 global 可作基准")
    per_pair = {}
    for mth in methods:
        sub = T[T["method"] == mth]
        vals = {}
        for (ds, sd), g in sub.groupby(["dataset", "seed"]):
            g = g.sort_values("marg_cov").drop_duplicates(subset=["marg_cov"])
            x = g["marg_cov"].to_numpy(float)
            cb = g["cov_cliffB"].to_numpy(float)
            wd = g["width"].to_numpy(float)
            if len(x) < 2 or x.min() > args.target or x.max() < args.target:
                continue
            vals[(ds, sd)] = (
                float(np.interp(args.target, x, cb)),
                float(np.interp(args.target, x, wd)),
            )
        per_pair[mth] = vals
        print(f"  {mth:16s} 可插值配对数 = {len(vals)}")

    ref = per_pair["global"]
    common = sorted(set(ref) & set(k for m in methods if m != "global" for k in per_pair[m]))
    rows = []
    for mth in methods:
        if mth == "global":
            continue
        d_cb, d_w = [], []
        for k in common:
            if k in per_pair[mth]:
                d_cb.append(per_pair[mth][k][0] - ref[k][0])
                d_w.append(per_pair[mth][k][1] - ref[k][1])
        d_cb = np.asarray(d_cb, float)
        d_w = np.asarray(d_w, float)
        if len(d_cb) < 5:
            continue
        w_p = _st.wilcoxon(d_cb, zero_method="wilcox").pvalue if np.any(d_cb != 0) else np.nan
        rows.append({
            "method": mth,
            "n_pairs": len(d_cb),
            "d_cliffB_mean": d_cb.mean(),
            "d_cliffB_pp": d_cb.mean() * 100,
            "wilcoxon_p": w_p,
            "pos_pairs": int((d_cb > 0).sum()),
            "d_width_mean": d_w.mean(),
        })
    R = pd.DataFrame(rows).sort_values("d_cliffB_pp", ascending=False)
    print(R.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # 逐数据集方向一致性（对头部方法）
    print("\n逐数据集方向一致性（悬崖组 Δ > 0 的数据集数）：")
    for mth in methods:
        if mth == "global":
            continue
        per_ds = {}
        for k in common:
            if k not in per_pair[mth]:
                continue
            v = per_pair[mth][k][0] - ref[k][0]
            per_ds.setdefault(k[0], []).append(v)
        if not per_ds:
            continue
        pos = sum(1 for v in per_ds.values() if np.mean(v) > 0)
        print(f"  {mth:16s} {pos}/{len(per_ds)}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        R.to_csv(os.path.join(args.out, "COVERAGE_MATCHED_PAIRED.csv"), index=False)
        print(f"\n[saved] {args.out}/COVERAGE_MATCHED_PAIRED.csv")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        S.to_csv(os.path.join(args.out, "COVERAGE_MATCHED_TRACE.csv"), index=False)
        I.to_csv(os.path.join(args.out, "COVERAGE_MATCHED.csv"), index=False)
        print(f"\n[saved] {args.out}/COVERAGE_MATCHED*.csv")


if __name__ == "__main__":
    main()
