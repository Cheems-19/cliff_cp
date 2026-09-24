#!/usr/bin/env python
"""Phase 1 结果分析：配对检验 + 跨数据集合并检验。

按 (dataset, seed) 配对比较，避免把数据集间的差异混进方法差异里。
对每个指标做三件事：
  1. 配对差值的均值与 95% CI
  2. Wilcoxon 符号秩检验
  3. 跨数据集的随机效应合并（每个数据集先取内部均值，再合并）

用法：
  python scripts/phase1_analyze.py --dir results_phase1
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from meta_analysis import dersimonian_lair  # noqa: E402


def paired_block(df, metric, base, cand):
    """返回按 (dataset, seed) 配对的差值序列。"""
    def pick(method):
        s = df[df["method"] == method].groupby(["dataset", "seed"])[metric].mean()
        return s.rename(method)

    return pd.concat([pick(base), pick(cand)], axis=1).dropna()


def report_pair(df, metric, base, cand, label):
    p = paired_block(df, metric, base, cand)
    if p.empty:
        return None
    d = (p[cand] - p[base]).to_numpy(float)
    se = d.std(ddof=1) / np.sqrt(d.size)
    try:
        w = stats.wilcoxon(d).pvalue if not np.allclose(d, 0) else np.nan
    except Exception:
        w = np.nan
    # 跨数据集合并：每个数据集内部的均值作为一个研究
    per_ds = (p[cand] - p[base]).groupby(level=0)
    eff = per_ds.mean().to_numpy(float)
    ses = (per_ds.std(ddof=1) / np.sqrt(per_ds.count())).to_numpy(float)
    dl = dersimonian_lair(eff, ses)
    return {
        "comparison": label,
        "metric": metric,
        "n_pairs": int(d.size),
        "delta_mean": float(d.mean()),
        "delta_ci_lo": float(d.mean() - 1.96 * se),
        "delta_ci_hi": float(d.mean() + 1.96 * se),
        "wilcoxon_p": float(w) if np.isfinite(w) else np.nan,
        "n_ds_delta_pos": int((eff > 0).sum()),
        "n_ds": int(eff.size),
        "meta_delta": dl.get("effect"),
        "meta_ci_lo": dl.get("ci_lo"),
        "meta_ci_hi": dl.get("ci_hi"),
        "meta_p": dl.get("p_value"),
        "meta_I2": dl.get("I2"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_phase1")
    args = ap.parse_args()
    d = pathlib.Path(args.dir)

    M = pd.read_csv(d / "METHODS.csv")
    DET = pd.read_csv(d / "DETECTORS.csv")

    # ---------------- 信号判别力 ----------------
    print("=" * 96)
    print("Q1  无标签信号对「落在区间外」的判别力（AUC）—— 12 数据集 × 10 划分")
    print("=" * 96)
    det = (
        DET.groupby("detector")
        .agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"), n=("auc", "size"))
        .reset_index()
    )
    # 每个数据集的均值 -> 跨数据集比较（避免大数据集主导）
    det_ds = (
        DET.groupby(["detector", "dataset"])["auc"].mean().reset_index()
        .groupby("detector")
        .agg(auc_ds_mean=("auc", "mean"), auc_ds_sd=("auc", "std"), n_ds=("auc", "size"))
        .reset_index()
    )
    det = det.merge(det_ds, on="detector").sort_values("auc_ds_mean", ascending=False)
    print(det[["detector", "auc_ds_mean", "auc_ds_sd", "n_ds"]].to_string(index=False))
    print(
        "\n参照：oracle 用测试标签（AUC 上界）；rf_std 是模型自带不确定性；\n"
        "d_* 是本项目的无标签数据侧信号；naive_proxy 是上一轮失败的信号。",
        flush=True,
    )

    # ---------------- 方法对比 ----------------
    print("\n" + "=" * 96)
    print("Q2/Q3  条件化校准：配对检验（+ = 相对全局 CP 的改进）")
    print("=" * 96)

    rows = []
    # 动态遍历所有 Mondrian 变体 —— 这样新增分区变量会自动进入消融表，
    # 不必每次改代码（分区变量消融是回答"结论是否只对某个信号成立"的关键）。
    cands = sorted(m for m in M["method"].unique() if m.startswith("mondrian_"))
    # 外部强对手（CQR）也一并放进对比表 —— 它不是"分区变量"，
    # 但必须出现在同一张表里，否则"分区有用/无用"的结论缺少参照。
    externals = [m for m in ("cqr",) if m in set(M["method"].unique())]
    cands = externals + cands
    print(f"\n[m] 共 {len(cands)} 个对比方法"
          f"（外部对手 {len(externals)} 个 + 分区变体 {len(cands)-len(externals)} 个）: "
          f"{', '.join(cands)}", flush=True)
    for cand in cands:
        for metric, label in [
            ("worst_d_wstd_cov", "最差组覆盖率(按标签离散度分箱)"),
            ("worst_rf_std_cov", "最差组覆盖率(按树间方差分箱)"),
            ("mean_width", "平均区间宽度(代价, - 更好)"),
        ]:
            sub = M[M["side"] == "__summary__"].copy()
            r = report_pair(sub, metric, "global", cand, label)
            if r:
                r["method"] = cand
                r["direction"] = "better" if "代价" not in label else "lower_is_better"
                rows.append(r)

        # 悬崖组覆盖率（机制定义）
        sub = M[(M["group_def"] == "cliffB_oracle") & (M["side"] == "cliff")]
        r = report_pair(sub, "coverage", "global", cand, "机制悬崖组覆盖率")
        if r:
            r["method"] = cand
            r["direction"] = "better"
            rows.append(r)

        # 边缘覆盖率（有效性不能牺牲）
        sub = M[M["side"] == "all"]
        r = report_pair(sub, "coverage", "global", cand, "边缘覆盖率(有效性)")
        if r:
            r["method"] = cand
            r["direction"] = "check~nominal"
            rows.append(r)

    R = pd.DataFrame(rows)[
        ["method", "comparison", "n_pairs", "delta_mean", "delta_ci_lo", "delta_ci_hi",
         "wilcoxon_p", "n_ds_delta_pos", "n_ds", "meta_delta", "meta_ci_lo",
         "meta_ci_hi", "meta_p", "meta_I2"]
    ]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    for m in cands:
        sub = R[R["method"] == m].drop(columns=["method"])
        if sub.empty:
            continue
        print(f"\n### {m}  vs  global")
        print(
            sub.to_string(
                index=False,
                formatters={
                    "delta_mean": "{:+.4f}".format,
                    "delta_ci_lo": "{:+.4f}".format,
                    "delta_ci_hi": "{:+.4f}".format,
                    "wilcoxon_p": "{:.2e}".format,
                    "meta_delta": "{:+.4f}".format,
                    "meta_ci_lo": "{:+.4f}".format,
                    "meta_ci_hi": "{:+.4f}".format,
                    "meta_p": "{:.2e}".format,
                    "meta_I2": "{:.1f}".format,
                },
            )
        )

    # ---------------- 消融一览 ----------------
    print("\n" + "=" * 96)
    print("分区变量消融一览（按悬崖组改善排序）")
    print("=" * 96)
    abl = R[R.comparison.str.contains("机制悬崖")][
        ["method", "delta_mean", "wilcoxon_p", "n_ds_delta_pos", "n_ds",
         "meta_delta", "meta_p", "meta_I2"]
    ].merge(
        R[R.comparison.str.contains("最差组覆盖率\\(按标签")][["method", "delta_mean"]]
        .rename(columns={"delta_mean": "worst_group_delta"}),
        on="method", how="left",
    ).merge(
        R[R.comparison.str.contains("代价")][["method", "delta_mean"]]
        .rename(columns={"delta_mean": "width_delta"}),
        on="method", how="left",
    ).sort_values("delta_mean", ascending=False)
    print(
        abl.to_string(
            index=False,
            formatters={
                "delta_mean": "{:+.4f}".format,
                "worst_group_delta": "{:+.4f}".format,
                "width_delta": "{:+.4f}".format,
                "wilcoxon_p": "{:.1e}".format,
                "meta_delta": "{:+.4f}".format,
                "meta_p": "{:.1e}".format,
                "meta_I2": "{:.0f}".format,
            },
        )
    )
    abl.to_csv(d / "PHASE1_ABLATION.csv", index=False)
    R.to_csv(d / "PHASE1_PAIRED.csv", index=False)
    det.to_csv(d / "PHASE1_DETECTORS_AGG.csv", index=False)
    print(f"\n[saved] {d}/PHASE1_PAIRED.csv, {d}/PHASE1_ABLATION.csv, {d}/PHASE1_DETECTORS_AGG.csv")

    # ---------------- 结论 ----------------
    print("\n" + "=" * 96)
    print("结论摘要")
    print("=" * 96)
    for m in cands:
        w = R[(R.method == m) & (R.comparison.str.contains("标签离散度"))]
        c = R[(R.method == m) & (R.comparison.str.contains("机制悬崖"))]
        mg = R[(R.method == m) & (R.comparison.str.contains("边缘"))]
        wd = R[(R.method == m) & (R.comparison.str.contains("代价"))]
        if not w.empty:
            print(
                f"{m:16s} 最差组 {w.iloc[0]['delta_mean']:+.4f} "
                f"(p={w.iloc[0]['wilcoxon_p']:.1e}) | "
                f"悬崖组 {c.iloc[0]['delta_mean']:+.4f} (p={c.iloc[0]['wilcoxon_p']:.1e}) | "
                f"边缘 {mg.iloc[0]['delta_mean']:+.4f} | "
                f"宽度 {wd.iloc[0]['delta_mean']:+.4f}"
            )


if __name__ == "__main__":
    main()
