#!/usr/bin/env python
"""Phase 3：分层核实（防 Simpson 悖论）+ 分箱趋势一致性的逐数据集检查。

为什么必须做
------------
Phase 2 的交叉表是把 15 个数据集的样本池在一起算的。池化比率存在 Simpson 悖论风险：
组内趋势可能与合并趋势相反。主图能不能直接进论文，取决于这个核实。

要核实的四个论断（逐数据集重算，看方向一致性）
  R1  信号"看见悬崖"的能力：cliffB 占比是否随分箱上升（rf_std 平缓、d_wstd 陡峭）
  R2  低分位箱的分位数被收紧（Δq < 0）、高分位箱被放宽（Δq > 0）
  R3  趋势方向：全局 CP 下 cliffB 覆盖率随 rf_std 分箱上升 —— 这是"模型自信处更危险"的证据
  R4  净效果：按 rf_std 条件化是否降低 cliffB 覆盖率、按 d_wstd 是否提高 —— 逐数据集符号一致性

用法: python scripts/phase3_stratified.py --dump results_phase1/dump --out results_phase3
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cliffcp import stats as S  # noqa: E402
from meta_analysis import dersimonian_lair  # noqa: E402

MIN_CLIFFB = 40  # 单数据集内 cliffB 太少则不做趋势判断


def load(dump_dir):
    fs = sorted(glob.glob(os.path.join(dump_dir, "*_molecules.csv.gz")))
    if not fs:
        raise SystemExit(f"no dump under {dump_dir}")
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    need = ["cliffB", "covered_global", "covered_rfstd", "covered_wstd",
            "bin_rfstd", "bin_wstd", "rf_std", "d_wstd", "q_global", "q_rfstd", "q_wstd"]
    d = d.dropna(subset=[c for c in need if c in d.columns]).reset_index(drop=True)
    print(f"[data] {len(fs)} datasets, {len(d):,} molecule records", flush=True)
    return d


def per_dataset(d):
    rows = []
    for ds, g in d.groupby("dataset"):
        n_cb = int(g["cliffB"].sum())
        if n_cb < MIN_CLIFFB:
            continue
        cb = g[g["cliffB"]]
        nb = g[~g["cliffB"]]

        # R2/R3：按 rf_std 分箱的趋势
        rf_share = g.groupby("bin_rfstd")["cliffB"].mean()
        rf_spear = stats.spearmanr(rf_share.index, rf_share.values)
        rf_dq_lo = float((g[g["bin_rfstd"] <= 1]["q_rfstd"] - g[g["bin_rfstd"] <= 1]["q_global"]).mean())
        rf_dq_hi = float((g[g["bin_rfstd"] >= 3]["q_rfstd"] - g[g["bin_rfstd"] >= 3]["q_global"]).mean())
        rf_cov_by_bin = cb.groupby("bin_rfstd")["covered_global"].mean()
        rf_cov_spear = stats.spearmanr(rf_cov_by_bin.index, rf_cov_by_bin.values)

        # d_wstd 分箱的对应量
        ws_share = g.groupby("bin_wstd")["cliffB"].mean()
        ws_spear = stats.spearmanr(ws_share.index, ws_share.values)
        ws_dq_hi = float((g[g["bin_wstd"] >= 3]["q_wstd"] - g[g["bin_wstd"] >= 3]["q_global"]).mean())

        # R4：净效果
        d_rf = float(cb["covered_rfstd"].mean() - cb["covered_global"].mean())
        d_ws = float(cb["covered_wstd"].mean() - cb["covered_global"].mean())

        rows.append({
            "dataset": ds,
            "n": len(g),
            "n_cliffB": n_cb,
            "cliffB_rate": n_cb / len(g),
            # 全局覆盖的整体水平
            "cov_all_global": float(g["covered_global"].mean()),
            "cov_cliffB_global": float(cb["covered_global"].mean()),
            "cov_noncliffB_global": float(nb["covered_global"].mean()),
            # R1 趋势
            "rf_share_spearman": float(rf_spear.statistic),
            "ws_share_spearman": float(ws_spear.statistic),
            "rf_share_bin0": float(rf_share.iloc[0]),
            "rf_share_bin4": float(rf_share.iloc[-1]),
            "ws_share_bin0": float(ws_share.iloc[0]),
            "ws_share_bin4": float(ws_share.iloc[-1]),
            # R2 分位数调整
            "rf_dq_lo_bins": rf_dq_lo,
            "rf_dq_hi_bins": rf_dq_hi,
            "ws_dq_hi_bins": ws_dq_hi,
            # R3 全局覆盖趋势
            "rf_cov_spearman": float(rf_cov_spear.statistic),
            # R4 净效果
            "delta_cliffB_rfstd": d_rf,
            "delta_cliffB_wstd": d_ws,
            "delta_noncliffB_rfstd": float(nb["covered_rfstd"].mean() - nb["covered_global"].mean()),
            "delta_noncliffB_wstd": float(nb["covered_wstd"].mean() - nb["covered_global"].mean()),
        })
    return pd.DataFrame(rows)


def verdict(t, d):
    print("=" * 100)
    print("逐数据集核对（R1-R4 的方向一致性）")
    print("=" * 100)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print(
        t[["dataset", "n", "n_cliffB", "cov_cliffB_global",
           "rf_share_spearman", "ws_share_spearman",
           "rf_dq_lo_bins", "rf_dq_hi_bins",
           "delta_cliffB_rfstd", "delta_cliffB_wstd"]].to_string(
            index=False,
            formatters={
                "cov_cliffB_global": "{:.3f}".format,
                "rf_share_spearman": "{:+.3f}".format,
                "ws_share_spearman": "{:+.3f}".format,
                "rf_dq_lo_bins": "{:+.3f}".format,
                "rf_dq_hi_bins": "{:+.3f}".format,
                "delta_cliffB_rfstd": "{:+.4f}".format,
                "delta_cliffB_wstd": "{:+.4f}".format,
            },
        )
    )

    n = len(t)
    print("\n" + "=" * 100)
    print("方向一致性统计（分子 = 方向正确的数据集数）")
    print("=" * 100)

    checks = [
        ("R1a rf_std 分箱 cliff 占比随箱上升",
         int((t["rf_share_spearman"] > 0).sum()), n),
        ("R1b d_wstd 分箱 cliff 占比随箱上升",
         int((t["ws_share_spearman"] > 0).sum()), n),
        ("R2a rf_std 在低分位箱收紧分位数 (Δq<0)",
         int((t["rf_dq_lo_bins"] < 0).sum()), n),
        ("R2b rf_std 在高分位箱放宽分位数 (Δq>0)",
         int((t["rf_dq_hi_bins"] > 0).sum()), n),
        ("R3  全局 CP 下 cliff 覆盖率随 rf_std 箱上升",
         int((t["rf_cov_spearman"] > 0).sum()), n),
        ("R4a rf_std 条件化降低 cliff 覆盖率",
         int((t["delta_cliffB_rfstd"] < 0).sum()), n),
        ("R4b d_wstd 条件化提高 cliff 覆盖率",
         int((t["delta_cliffB_wstd"] > 0).sum()), n),
    ]
    for name, k, tot in checks:
        bar = "#" * int(round(20 * k / tot))
        print(f"  {name:46s} {k:3d}/{tot}  {bar}")

    print("\n" + "=" * 100)
    print("跨数据集合并（DerSimonian–Laird 随机效应，按种子配对后取每个数据集的效应量）")
    print("=" * 100)
    print(
        "  说明：先在同一 (数据集, 种子) 内做配对差（同一批测试点上比较两方法），\n"
        "        再在数据集内跨种子求均值与标准误，最后跨数据集合并。\n"
        "        直接对池化覆盖率做二项近似会严重低估功效（配对信息被丢掉）。",
        flush=True,
    )
    for cand, label in [
        ("rfstd", "rf_std 条件化对 cliff 覆盖率"),
        ("wstd", "d_wstd 条件化对 cliff 覆盖率"),
    ]:
        eff, ses, nds, npos = [], [], 0, 0
        for ds, g in d.groupby("dataset"):
            cb = g[g["cliffB"]]
            per_seed = (
                cb.groupby("seed")[[f"covered_{cand}", "covered_global"]].mean()
            )
            if len(per_seed) < 3:
                continue
            delta = per_seed[f"covered_{cand}"] - per_seed["covered_global"]
            eff.append(float(delta.mean()))
            ses.append(float(delta.std(ddof=1) / np.sqrt(len(delta))))
            nds += 1
            npos += int(delta.mean() > 0)
        dl = dersimonian_lair(np.asarray(eff), np.asarray(ses))
        print(
            f"  {label:30s} 合并 = {dl.get('effect', float('nan')):+.4f} "
            f"[{dl.get('ci_lo', float('nan')):+.4f}, {dl.get('ci_hi', float('nan')):+.4f}]  "
            f"p = {dl.get('p_value', float('nan')):.2e}  I2 = {dl.get('I2', 0):.0f}%  "
            f"正向数据集 {npos}/{nds}",
            flush=True,
        )

    # Simpson 专项：箱内趋势 vs 合并趋势
    print("\n" + "=" * 100)
    print("Simpson 悖论专项：R3（全局 CP 下 cliff 覆盖率随 rf_std 分箱上升）是否在单数据集内成立")
    print("=" * 100)
    pos = int((t["rf_cov_spearman"] > 0).sum())
    print(
        f"  合并数据上该趋势为正；单数据集内 {pos}/{n} 也为正。"
        f"{'趋势稳健，可放心合并报。' if pos >= 0.7 * n else '注意：趋势在部分数据集内不成立，合并结论需谨慎表述。'}"
    )

    # 按数据集分层重算"低分位箱的 cliff 覆盖率崩塌"
    print("\n" + "=" * 100)
    print("Simpson 专项：'低分位箱 cliff 覆盖率崩塌'是否在单数据集内成立")
    print("=" * 100)
    return t


def bin_level(d):
    rows = []
    for ds, g in d.groupby("dataset"):
        if int(g["cliffB"].sum()) < MIN_CLIFFB:
            continue
        cb = g[g["cliffB"]]
        lo = cb[cb["bin_rfstd"] <= 1]
        if len(lo) >= 20:
            rows.append({
                "dataset": ds,
                "n_cliffB_low_bins": len(lo),
                "cov_lo_global": float(lo["covered_global"].mean()),
                "cov_lo_rfstd": float(lo["covered_rfstd"].mean()),
                "delta_lo": float(lo["covered_rfstd"].mean() - lo["covered_global"].mean()),
            })
    b = pd.DataFrame(rows)
    if b.empty:
        return b
    print(
        b.to_string(
            index=False,
            formatters={
                "cov_lo_global": "{:.3f}".format,
                "cov_lo_rfstd": "{:.3f}".format,
                "delta_lo": "{:+.4f}".format,
            },
        )
    )
    print(
        f"\n  {int((b['delta_lo'] < 0).sum())}/{len(b)} 个数据集的低分位箱 cliff 覆盖率"
        f"在换用 rf_std 分区后下降（合并均值 {b['delta_lo'].mean():+.4f}）"
    )
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="results_phase1/dump")
    ap.add_argument("--out", default="results_phase3")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(args.dump)
    t = per_dataset(d)
    verdict(t, d)
    print()
    b = bin_level(d)

    t.to_csv(out / "STRATIFIED_PER_DATASET.csv", index=False)
    if not b.empty:
        b.to_csv(out / "STRATIFIED_LOWBIN.csv", index=False)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
