#!/usr/bin/env python
"""在已有分子级 dump 上离线探索"更好的校准规则"。

动机
----
Phase 2 的诊断为：模型侧信号（rf_std）能预测失效（AUC 0.699）却看不见悬崖（AUC 0.572），
错配量 +0.126；于是按 rf_std 做 Mondrian 分区时，**被收紧的分箱恰好是悬崖藏身处**，
悬崖组覆盖率从 0.723 掉到 0.669（15/15 数据集一致）。

由此自然产生一个修复假设：
    只要禁止任何分箱的区间比"全局"更窄，就能保住 rf_std 的收益、去掉它的伤害。
即 q_rule = max(q_bin, q_global)。

本脚本用 dump 里已存的每分子 q 值直接评估该规则，**不需重跑流水线**。
区间为对称形式（lower = pred - q, upper = pred + q），因此
    covered == (|y - pred| <= q)
脚本自带一致性自检，验证这一点在 dump 上成立。

用法:
    python scripts/rule_explore.py --dump results_ecfp_abl/dump --out results_ecfp_abl
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats


# ---------------- 规则定义 ----------------
# 每个规则 = 从 dump 的列算出每分子的 q。全部是"仅用 q 值组合"的操作。
RULES = {
    "global":        lambda d: d["q_global"],
    "rfstd":         lambda d: d["q_rfstd"],
    "wstd":          lambda d: d["q_wstd"],
    "std":           lambda d: d["q_std"],
    "iqr":           lambda d: d["q_iqr"],
    "nb":            lambda d: d["q_nb"],
    "nnsim":         lambda d: d["q_nnsim"],
    # —— 修复假设 ——
    "rfstd_floor_g": lambda d: np.maximum(d["q_rfstd"], d["q_global"]),
    "rfstd_floor_ws": lambda d: np.maximum(d["q_rfstd"], d["q_wstd"]),
    "rfstd_avg_ws":  lambda d: 0.5 * (d["q_rfstd"] + d["q_wstd"]),
    "rfstd_floor_g_ws": lambda d: np.maximum(np.maximum(d["q_rfstd"], d["q_global"]), d["q_wstd"]),
    # —— 反向对照：更激进的收紧（预期有害）——
    "rfstd_min_g":   lambda d: np.minimum(d["q_rfstd"], d["q_global"]),
}

# 哪些规则是"新提出"的，报告时分开展示
FIXED_RULES = ["rfstd_floor_g", "rfstd_floor_ws", "rfstd_avg_ws",
               "rfstd_floor_g_ws", "rfstd_min_g"]


def load_dump(dump_dir: str):
    files = sorted(glob.glob(os.path.join(dump_dir, "*.csv.gz")))
    if not files:
        raise SystemExit(f"没有找到 dump 文件: {dump_dir}/*.csv.gz")
    frames = [pd.read_csv(f) for f in files]
    D = pd.concat(frames, ignore_index=True)
    return D


def self_check(D: pd.DataFrame):
    """验证 covered_* == (|y-pred| <= q_*)；这是整个离线模拟的地基。"""
    res = abs(D["y"] - D["pred"])
    print("自检：covered_* 是否等于 |y-pred| <= q_*")
    worst = 0.0
    for tag in ["global", "rfstd", "wstd", "std", "iqr", "nb", "nnsim"]:
        cc, qc = f"covered_{tag}", f"q_{tag}"
        if cc not in D.columns or qc not in D.columns:
            continue
        mine = (res <= D[qc].to_numpy(float)).astype(int)
        theirs = D[cc].to_numpy(int)
        mism = float(np.mean(mine != theirs))
        worst = max(worst, mism)
        print(f"  {tag:8s} 不一致率 = {mism*100:.4f}%")
    if worst > 1e-6:
        print("  !! 自检未通过 —— 说明 dump 的区间不是对称形式，"
              "或 covered 列与 q 列不同源，离线模拟结论不可信")
    else:
        print("  自检通过（不一致率 0），可用 q 值离线模拟其他规则\n")
    return worst


def worst_group_coverage(cov: np.ndarray, bins: np.ndarray, min_n: int = 10):
    """按分箱取覆盖率，返回 (最差组覆盖率, 组间极差)。忽略样本过少的组。"""
    vals, spreads = [], []
    for b in np.unique(bins):
        m = bins == b
        if int(m.sum()) < min_n:
            continue
        vals.append(float(cov[m].mean()))
    if len(vals) < 2:
        return np.nan, np.nan
    return float(np.min(vals)), float(max(vals) - min(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    D = load_dump(args.dump)
    print(f"载入 {len(D):,} 条分子级记录，"
          f"{D['dataset'].nunique()} 数据集，{D['seed'].nunique()} 种子\n")
    self_check(D)

    res = np.abs(D["y"].to_numpy(float) - D["pred"].to_numpy(float))
    bin_ws = D["bin_wstd"].to_numpy(int)
    bin_rf = D["bin_rfstd"].to_numpy(int)
    cliffA = D["cliffA"].to_numpy(bool)
    cliffB = D["cliffB"].to_numpy(bool)

    # 逐 (dataset, seed) 计算指标
    rows = []
    keys = ["dataset", "seed"]
    grp_idx = list(D.groupby(keys, sort=False).indices.items())
    for tag, fn in RULES.items():
        q = np.asarray(fn(D), float)
        q = np.where(np.isfinite(q), q, np.nan)
        cov = (res <= q).astype(float)
        cov = np.where(np.isfinite(q), cov, np.nan)
        for (ds, sd), idx in grp_idx:
            c = cov[idx]
            ca, cb = cliffA[idx], cliffB[idx]
            ok = np.isfinite(c)
            if ok.sum() < 30:
                continue
            c2 = c[ok]
            w, sp = worst_group_coverage(c2, bin_ws[idx][ok])
            w2, sp2 = worst_group_coverage(c2, bin_rf[idx][ok])
            rows.append({
                "rule": tag, "dataset": ds, "seed": sd,
                "n": int(ok.sum()),
                "coverage": float(c2.mean()),
                "cov_cliffA": float(c2[ca[ok]].mean()) if ca[ok].sum() else np.nan,
                "cov_cliffB": float(c2[cb[ok]].mean()) if cb[ok].sum() else np.nan,
                "n_cliffB": int(cb[ok].sum()),
                "cov_noncliffB": float(c2[~cb[ok]].mean()) if (~cb[ok]).sum() else np.nan,
                "worst_ws": w, "spread_ws": sp,
                "worst_rf": w2, "spread_rf": sp2,
                "width": float(2.0 * np.nanmean(q[idx][ok])),
            })
    R = pd.DataFrame(rows)

    # ---------------- 汇总 ----------------
    print("=" * 118)
    print("规则总体表现（对 (dataset, seed) 取均值）")
    print("=" * 118)
    agg = R.groupby("rule").agg(
        coverage=("coverage", "mean"),
        cov_cliffA=("cov_cliffA", "mean"),
        cov_cliffB=("cov_cliffB", "mean"),
        worst_ws=("worst_ws", "mean"),
        spread_ws=("spread_ws", "mean"),
        worst_rf=("worst_rf", "mean"),
        width=("width", "mean"),
    ).reset_index()
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------- 与 global 的配对差异 ----------------
    print("\n" + "=" * 118)
    print("相对 global 的配对差异（Wilcoxon 符号秩；逐 (dataset, seed) 配对）")
    print("=" * 118)
    base = R[R["rule"] == "global"].set_index(keys)
    out = []
    for tag in RULES:
        if tag == "global":
            continue
        sub = R[R["rule"] == tag].set_index(keys)
        j = sub.join(base, lsuffix="_r", rsuffix="_b", how="inner")
        rec = {"rule": tag, "n_pairs": len(j)}
        for col, lab in [("cov_cliffB", "d_cliffB"), ("worst_ws", "d_worst"),
                         ("coverage", "d_cov"), ("width", "d_width")]:
            a = j[f"{col}_r"].to_numpy(float)
            b = j[f"{col}_b"].to_numpy(float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 6:
                rec[lab] = np.nan
                rec[lab + "_p"] = np.nan
                rec[lab + "_pos"] = np.nan
                continue
            diff = a[m] - b[m]
            rec[lab] = float(diff.mean())
            try:
                rec[lab + "_p"] = float(stats.wilcoxon(diff).pvalue) if not np.allclose(diff, 0) else np.nan
            except Exception:
                rec[lab + "_p"] = np.nan
            # 按数据集方向一致性
            d2 = pd.DataFrame({"dataset": j.index.get_level_values(0)[m], "d": diff})
            dd = d2.groupby("dataset")["d"].mean()
            rec[lab + "_pos"] = f"{int((dd>0).sum())}/{len(dd)}"
        out.append(rec)
    P = pd.DataFrame(out)
    P = P.sort_values("d_cliffB", ascending=False)
    print(P.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # ---------------- 关键规则的分箱剖面 ----------------
    print("\n" + "=" * 118)
    print("关键规则：按 rf_std 分箱的覆盖率剖面（看得见'收紧发生在哪'）")
    print("=" * 118)
    prof_rows = []
    for tag in ["global", "rfstd", "wstd", "rfstd_floor_g", "rfstd_floor_ws"]:
        if tag not in RULES:
            continue
        q = np.asarray(RULES[tag](D), float)
        cov = (res <= q).astype(float)
        for b in sorted(np.unique(bin_rf)):
            m = (bin_rf == b) & np.isfinite(cov)
            if m.sum() < 30:
                continue
            prof_rows.append({
                "rule": tag, "bin_rfstd": int(b),
                "n": int(m.sum()),
                "frac_cliffB": float(cliffB[m].mean()) * 100,
                "cov_all": float(cov[m].mean()),
                "cov_cliffB": float(cov[m & cliffB].mean()) if (m & cliffB).sum() else np.nan,
                "cov_noncliffB": float(cov[m & ~cliffB].mean()) if (m & ~cliffB).sum() else np.nan,
                "mean_q": float(np.nanmean(q[m])),
            })
    prof = pd.DataFrame(prof_rows)
    for tag, sub in prof.groupby("rule"):
        print(f"\n--- {tag} ---")
        print(sub.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        R.to_csv(os.path.join(args.out, "RULE_PER_PAIR.csv"), index=False)
        agg.to_csv(os.path.join(args.out, "RULE_AGG.csv"), index=False)
        P.to_csv(os.path.join(args.out, "RULE_PAIRED.csv"), index=False)
        prof.to_csv(os.path.join(args.out, "RULE_PROFILE.csv"), index=False)
        print(f"\n[saved] {args.out}/RULE_*.csv")


if __name__ == "__main__":
    main()
