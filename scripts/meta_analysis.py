#!/usr/bin/env python
"""跨数据集的合并检验（meta-analysis）。

为什么必须做：单个 MoleculeACE 数据集的悬崖分子只有 ~250 个，
测试集里只剩 40–60 个，再按效力分箱做分层置换检验后每层仅约 10 个样本，
功效极低 —— 会出现"效应方向 8/8 一致、但逐个 p 都在 0.05 上方"的现象。
这时逐数据集判决会得出"无效应"的错误结论。正确做法是合并证据。

两种合并方式，互补使用：
  1. 效应量合并（DerSimonian–Laird 随机效应）——给出合并 gap 及其置信区间
  2. p 值合并（Fisher / Stouffer）——给出整体显著性

用法：
  python scripts/meta_analysis.py --gaps results_server/GAPS.csv
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats


def fisher_combine(pvals):
    p = np.asarray([x for x in pvals if np.isfinite(x) and x > 0], dtype=float)
    if p.size == 0:
        return np.nan, np.nan
    chi2 = -2.0 * np.log(p).sum()
    dof = 2 * p.size
    return float(chi2), float(stats.chi2.sf(chi2, dof))


def stouffer_combine(pvals, signs):
    p = np.asarray(pvals, dtype=float)
    s = np.asarray(signs, dtype=float)
    m = np.isfinite(p) & (p > 0) & (p < 1)
    p, s = p[m], s[m]
    if p.size == 0:
        return np.nan, np.nan
    z = stats.norm.isf(p / 2.0) * np.sign(s)
    Z = z.sum() / np.sqrt(p.size)
    return float(Z), float(2 * stats.norm.sf(abs(Z)))


def dersimonian_lair(effects, ses):
    """随机效应合并（DerSimonian–Laird）。返回合并效应、SE、z、p、I^2。"""
    y = np.asarray(effects, float)
    se = np.asarray(ses, float)
    m = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y, se = y[m], se[m]
    k = y.size
    if k == 0:
        return dict(k=0)
    w = 1.0 / se**2
    y_fixed = (w * y).sum() / w.sum()
    Q = (w * (y - y_fixed) ** 2).sum()
    dfree = k - 1
    C = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0.0, (Q - dfree) / C) if C > 0 else 0.0
    wr = 1.0 / (se**2 + tau2)
    y_pool = (wr * y).sum() / wr.sum()
    se_pool = float(np.sqrt(1.0 / wr.sum()))
    zz = y_pool / se_pool
    p = float(2 * stats.norm.sf(abs(zz)))
    i2 = float(max(0.0, (Q - dfree) / Q) * 100) if Q > 0 else 0.0
    return dict(
        k=int(k), effect=float(y_pool), se=se_pool,
        ci_lo=float(y_pool - 1.96 * se_pool), ci_hi=float(y_pool + 1.96 * se_pool),
        z=float(zz), p_value=p, tau2=float(tau2), I2=i2, Q=float(Q),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", required=True)
    ap.add_argument("--min-seeds", type=int, default=5)
    args = ap.parse_args()

    g = pd.read_csv(args.gaps)
    g = g[g["n_seeds"] >= args.min_seeds]
    if g.empty:
        print("no rows after filtering")
        return 1

    rows = []
    for (gd, mt), d in g.groupby(["group_def", "method"]):
        d = d[np.isfinite(d["gap_mean"])]
        if len(d) < 2:
            continue
        ses = d["gap_sd"].to_numpy(float) / np.sqrt(d["n_seeds"].to_numpy(float))
        ses = np.where(np.isfinite(ses) & (ses > 0), ses, np.nan)
        dl = dersimonian_lair(d["gap_mean"], ses)

        # 用每个数据集的 median p 作为该数据集的代表 p，避免同数据集内多种子相关
        p_med = float(np.nanmedian(d["perm_p_mean"]))
        p_all = d["perm_p_mean"].to_numpy(float)
        signs = np.sign(d["gap_mean"].to_numpy(float))
        chi2, p_fisher_med = fisher_combine([p_med])
        _, p_fisher_all = fisher_combine(p_all)
        Z, p_stouffer = stouffer_combine(p_all, signs)

        rows.append(
            {
                "group_def": gd,
                "method": mt,
                "n_datasets": len(d),
                "n_datasets_gap_pos": int((d["gap_mean"] > 0).sum()),
                "gap_fixed_mean": float(np.nanmean(d["gap_mean"])),
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in dl.items()},
                "fisher_p_all_seeds": p_fisher_all,
                "stouffer_p": p_stouffer,
                "stouffer_Z": round(float(Z), 3) if np.isfinite(Z) else None,
                "min_perm_p": float(np.nanmin(p_all)),
                "median_perm_p": p_med,
            }
        )

    out = pd.DataFrame(rows).sort_values(["group_def", "method"])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print("=" * 100)
    print("跨数据集合并检验")
    print("=" * 100)
    for gd in ["defA", "defB_oracle", "defC_proxy"]:
        sub = out[out.group_def == gd]
        if sub.empty:
            continue
        print(f"\n### {gd}")
        print(
            sub[
                [
                    "method", "n_datasets", "n_datasets_gap_pos",
                    "gap_fixed_mean", "effect", "ci_lo", "ci_hi",
                    "p_value", "stouffer_p", "fisher_p_all_seeds", "I2",
                ]
            ].to_string(index=False)
        )

    print("\n" + "=" * 100)
    print("判决（以 defA + global 为主检验，defB 为机制验证）")
    print("=" * 100)
    for gd in ["defA", "defB_oracle"]:
        r = out[(out.group_def == gd) & (out.method == "global")]
        if r.empty:
            continue
        r = r.iloc[0]
        sig = (r["p_value"] < 0.05) and (r["ci_lo"] > 0)
        print(
            f"{gd:14s} 合并 gap={r['effect']:.4f} "
            f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]  p={r['p_value']:.5f}  "
            f"方向一致 {r['n_datasets_gap_pos']}/{r['n_datasets']}  "
            f"I2={r['I2']:.1f}%  -> {'显著' if sig else '不显著'}"
        )

    out.to_csv("results_server/META.csv", index=False)
    print("\n[saved] results_server/META.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
