#!/usr/bin/env python
"""校准规则的"效率前沿"分析：在**匹配宽度**下比较各规则。

动机
----
rule_explore.py 发现 q = max(q_rfstd, q_global)（"保守下界 / 不下缩"规则）
能把悬崖组覆盖率拉回 +4.8 pp（15/15），但平均宽度涨 +10.8%。
审稿人一定会问：这不就是"变宽了当然覆盖更好"吗？

本脚本用两种方式回答：
  1) **匹配宽度归一化**：把每个规则的 q 整体乘一个常数 c，使该 (dataset, seed) 的
     平均宽度等于 global 的平均宽度，再重算覆盖率。这样比较的是"同等代价下的收益"。
     （注意：这是诊断性归一化，不是合法的共形流程，只用来分离"效率"与"保守性"。）
  2) **γ 连续扫描**：q = max(q_rfstd, γ·q_global)，γ ∈ [0,1]，
     描出一条"宽度代价 ↔ 悬崖组覆盖"的前沿曲线。

同时给出：rfstd 那条"更窄且最差组更好"的曲线，在匹配宽度上是否仍成立。

用法:
    python scripts/rule_frontier.py --dump results_ecfp_abl/dump --out results_ecfp_abl
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats


def load_dump(dump_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(dump_dir, "*.csv.gz")))
    if not files:
        raise SystemExit(f"没有找到 dump: {dump_dir}/*.csv.gz")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def build_rules(D, gammas):
    """返回 {规则名: q 数组}。所有规则都只由 dump 里已存的 q 列组合而成。"""
    g = D["q_global"].to_numpy(float)
    rf = D["q_rfstd"].to_numpy(float)
    ws = D["q_wstd"].to_numpy(float)
    R = {
        "global": g,
        "rfstd": rf,
        "wstd": ws,
        "std": D["q_std"].to_numpy(float),
        "iqr": D["q_iqr"].to_numpy(float),
        "nnsim": D["q_nnsim"].to_numpy(float),
        "nb": D["q_nb"].to_numpy(float),
    }
    for gm in gammas:
        R[f"floor_g{gm:.2f}"] = np.maximum(rf, gm * g)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()

    D = load_dump(args.dump)
    gammas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    Q = build_rules(D, gammas)
    print(f"载入 {len(D):,} 行 | {D['dataset'].nunique()} 数据集 "
          f"| {D['seed'].nunique()} 种子 | {len(Q)} 个规则\n")

    res = np.abs(D["y"].to_numpy(float) - D["pred"].to_numpy(float))
    cliffA = D["cliffA"].to_numpy(bool)
    cliffB = D["cliffB"].to_numpy(bool)
    # 分组键
    key = D["dataset"].astype(str) + "|" + D["seed"].astype(str)
    key_codes, key_uniq = pd.factorize(key)
    nk = len(key_uniq)

    rows = []
    for tag, q in Q.items():
        q = np.asarray(q, float)
        ok = np.isfinite(q)
        cov = np.where(ok, (res <= q).astype(float), np.nan)
        # 逐 (dataset, seed) 的宽度，用于匹配归一化
        wsum = np.bincount(key_codes[ok], weights=q[ok], minlength=nk)
        wcnt = np.bincount(key_codes[ok], minlength=nk)
        w_pair = np.where(wcnt > 0, wsum / np.maximum(wcnt, 1), np.nan)
        rows.append({"rule": tag, "_q": q, "_cov": cov, "_w_pair": w_pair,
                     "_ok": ok})
    # global 的配对宽度，作为归一化基准
    w_ref = rows[0]["_w_pair"]

    out = []
    for r in rows:
        tag, q, cov, w_pair, ok = r["rule"], r["_q"], r["_cov"], r["_w_pair"], r["_ok"]
        c_pair = np.where(w_pair > 0, w_ref / np.where(w_pair > 0, w_pair, 1), np.nan)
        c = c_pair[key_codes]
        cov_m = np.where(ok & np.isfinite(c), (res <= q * c).astype(float), np.nan)

        def agg(cc, name):
            d = {"rule": tag, "view": name}
            n = np.isfinite(cc)
            d["coverage"] = float(np.nanmean(cc))
            d["cov_cliffA"] = float(np.nanmean(cc[cliffA & n]))
            d["cov_cliffB"] = float(np.nanmean(cc[cliffB & n]))
            d["n_cliffB"] = int((cliffB & n).sum())
            d["width"] = float(2.0 * np.nanmean(q[ok])) if name == "raw" else float(
                2.0 * np.nanmean((q * c)[ok & np.isfinite(c)]))
            return d

        out.append(agg(cov, "raw"))
        out.append(agg(cov_m, "matched_width"))

    A = pd.DataFrame(out)
    print("=" * 108)
    print("原始视图（未归一化）")
    print("=" * 108)
    print(A[A.view == "raw"].drop(columns=["view"]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n" + "=" * 108)
    print("匹配宽度视图（每个规则的 q 整体缩放，使平均宽度 = global）")
    print("   —— 这才是「同等代价下的收益」，排除「变宽当然覆盖更好」这个平凡解释")
    print("=" * 108)
    print(A[A.view == "matched_width"].drop(columns=["view"]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------- 逐配对统计（原始视图）----------------
    print("\n" + "=" * 108)
    print("相对 global 的配对差异：原始 vs 匹配宽度（Wilcoxon，n_pairs = 数据集×种子）")
    print("=" * 108)
    rec = []
    base_b = None
    for r in rows:
        tag, q, cov, w_pair, ok = r["rule"], r["_q"], r["_cov"], r["_w_pair"], r["_ok"]
        # 逐配对指标
        def pairwise(cc):
            cb = np.bincount(key_codes[cliffB & np.isfinite(cc)],
                             weights=cc[cliffB & np.isfinite(cc)], minlength=nk)
            cn = np.bincount(key_codes[cliffB & np.isfinite(cc)], minlength=nk)
            return np.where(cn > 0, cb / np.maximum(cn, 1), np.nan)
        cb_raw = pairwise(cov)
        c_pair = np.where(w_pair > 0, w_ref / np.where(w_pair > 0, w_pair, 1), np.nan)
        c = c_pair[key_codes]
        cov_m = np.where(ok & np.isfinite(c), (res <= q * c).astype(float), np.nan)
        cb_m = pairwise(cov_m)
        if tag == "global":
            base_b, base_m = cb_raw, cb_m
        rec.append({"rule": tag, "_raw": cb_raw, "_m": cb_m,
                    "_w": 2 * w_pair})
    base_b = rec[0]["_raw"]
    base_m = rec[0]["_m"]
    base_w = rec[0]["_w"]
    tab = []
    for r in rec:
        d = {"rule": r["rule"]}
        for suff, a, b in [("raw", r["_raw"], base_b), ("mtd", r["_m"], base_m)]:
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 6:
                continue
            diff = a[m] - b[m]
            try:
                p = float(stats.wilcoxon(diff).pvalue) if not np.allclose(diff, 0) else np.nan
            except Exception:
                p = np.nan
            d[f"d_cliff_{suff}"] = float(diff.mean())
            d[f"p_{suff}"] = p
            d[f"pos_{suff}"] = f"{int((diff>0).sum())}/{len(diff)}"
        m = np.isfinite(r["_w"]) & np.isfinite(base_w)
        d["d_width"] = float((r["_w"][m] - base_w[m]).mean())
        d["w_ratio"] = float((r["_w"][m] / np.maximum(base_w[m], 1e-9)).mean())
        tab.append(d)
    T = pd.DataFrame(tab)
    print(T.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print("\n" + "=" * 108)
    print("γ 扫描前沿：q = max(q_rfstd, γ·q_global)")
    print("  γ=0 即纯 rfstd（最窄、最伤悬崖）；γ=1 即完整的「不下缩」规则")
    print("=" * 108)
    front = T[T.rule.str.startswith("floor_g")].copy()
    front["gamma"] = front.rule.str.replace("floor_g", "", regex=False).astype(float)
    front = front.sort_values("gamma")
    print(front[["gamma", "d_cliff_raw", "pos_raw", "d_cliff_mtd", "pos_mtd",
                 "d_width", "w_ratio", "p_raw"]].to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        A.to_csv(os.path.join(args.out, "FRONTIER_RULES.csv"), index=False)
        T.to_csv(os.path.join(args.out, "FRONTIER_PAIRED.csv"), index=False)
        front.to_csv(os.path.join(args.out, "FRONTIER_GAMMA.csv"), index=False)
        print(f"\n[saved] {args.out}/FRONTIER_*.csv")


if __name__ == "__main__":
    main()
