"""Phase 6b：AD 阈值敏感性 + 第二套 AD 代理 —— 那两个核心数字稳不稳？

Phase 6 的核心数字是"61.8% 的失效发生在 AD 内部"与"域内悬崖失败率 28.45% > 域外 19.26%"。
但"AD 内"是按最近邻相似度的 20% 分位切的，这个阈值是任意的。审稿人一定会问：
换个阈值结论还在不在？换个 AD 代理（不是相似度）结论还在不在？

本脚本做两件事：
  1. 阈值扫描：AD 内界定改 10% / 20% / 30% / 40% 分位
  2. 换代理：距离型（1-NN Tanimoto 相似度）vs 密度型（训练邻居数 d_nb）
     —— 两族都是教科书上的标准 AD 方法

并对"域内悬崖失败率 > 域外失败率"这条最刺眼的结论报**逐数据集方向一致性**。

用法:
    python scripts/phase6b_ad_sensitivity.py --dump results_ecfp_abl/dump --out results_ecfp_abl
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cliffcp import stats as S  # noqa: E402


def load(dump_dir):
    fs = sorted(glob.glob(os.path.join(dump_dir, "*.csv.gz")))
    if not fs:
        raise SystemExit(f"没找到 {dump_dir}/*.csv.gz")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def recall_at(scores, is_fail, frac):
    n = len(scores)
    k = int(round(frac * n))
    if k < 1 or is_fail.sum() == 0:
        return np.nan
    thr = np.sort(scores)[::-1][k - 1]
    return float(is_fail[scores >= thr].sum() / is_fail.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--flag-frac", type=float, default=0.20)
    args = ap.parse_args()

    D = load(args.dump)
    fail = ~D["covered_global"].to_numpy(bool)
    cliff = D["cliffB"].to_numpy(bool)
    wstd = D["d_wstd"].to_numpy(float)

    # 组内秩分位（跨数据集可比）
    gcodes = pd.factorize(D["dataset"].astype(str) + "|" + D["seed"].astype(str))[0]
    print(f"载入 {len(D):,} 行 | {D['dataset'].nunique()} 数据集 | "
          f"{D['seed'].nunique()} 种子 | 全局欠覆盖 {fail.mean():.4f}\n")

    # 两个 AD 代理：都换成"越高越像在域内"的组内分位
    #   1) 距离型：1-NN Tanimoto 相似度。NaN（无邻居）显式当作最疏
    nnsim = D["d_nn_sim"].to_numpy(float)
    n_nan = int((~np.isfinite(nnsim)).sum())
    q_sim = S.group_rank_quantile(np.where(np.isfinite(nnsim), nnsim, 0.0), gcodes)
    #   2) 密度型：训练邻居数。NaN 同样当作最少
    nb = D["d_nb"].to_numpy(float)
    q_nb = S.group_rank_quantile(np.where(np.isfinite(nb), nb, 0.0), gcodes)
    #   预警用的数据侧信号
    q_ws = S.group_rank_quantile(np.where(np.isfinite(wstd), wstd, np.nan), gcodes)
    q_ws = np.where(np.isfinite(q_ws), q_ws, 0.0)
    print(f"注意：{n_nan} 条记录的 1-NN 相似度为 NaN（无匹配邻居），已按最疏处理\n")

    proxies = {"距离型 AD（1-NN 相似度）": q_sim, "密度型 AD（训练邻居数）": q_nb}
    thr_list = [0.10, 0.20, 0.30, 0.40]

    all_rows, ds_rows = [], []
    for pname, q in proxies.items():
        for thr in thr_list:
            in_ad = q >= thr
            out_ad = ~in_ad
            f_in, f_out = fail[in_ad], fail[out_ad]
            c_in_cl, c_in_non = fail[in_ad & cliff], fail[in_ad & ~cliff]

            row = {
                "proxy": pname,
                "ad_quantile": thr,
                "n_in_ad": int(in_ad.sum()),
                "share_of_failures_in_ad": (f_in.sum() / fail.sum()) * 100,
                "fail_rate_in_ad": f_in.mean(),
                "fail_rate_out_ad": f_out.mean(),
                "cov_in_ad_cliff": 1 - c_in_cl.mean() if c_in_cl.size else np.nan,
                "fail_in_ad_cliff": c_in_cl.mean() if c_in_cl.size else np.nan,
                "n_in_ad_cliff": int((in_ad & cliff).sum()),
                "cov_in_ad_noncliff": 1 - c_in_non.mean() if c_in_non.size else np.nan,
                "n_cliff_out_ad": int((out_ad & cliff).sum()),
            }
            # 域内预警召回
            num = den = 0.0
            for gc in np.unique(gcodes):
                m = gcodes == gc
                mm = m & in_ad
                if mm.sum() < 20 or fail[mm].sum() == 0:
                    continue
                r = recall_at(q_ws[mm], fail[mm], args.flag_frac)
                if np.isfinite(r):
                    num += r * fail[mm].sum()
                    den += fail[mm].sum()
            row["in_ad_recall"] = num / den if den else np.nan
            row["in_ad_recall_lift"] = row["in_ad_recall"] / args.flag_frac if den else np.nan
            all_rows.append(row)

            # 逐 (dataset, seed) 的"域内悬崖失败率 − 域外失败率"，再按数据集汇总方向
            diffs = {}
            for gc in np.unique(gcodes):
                m = gcodes == gc
                a = fail[m & in_ad & cliff]
                b = fail[m & out_ad]
                if a.size < 5 or b.size < 20:
                    continue
                dsname = D["dataset"].to_numpy()[m][0]
                diffs.setdefault(dsname, []).append(a.mean() - b.mean())
            if diffs:
                per_ds = {k: float(np.mean(v)) for k, v in diffs.items()}
                pos = sum(1 for v in per_ds.values() if v > 0)
                ds_rows.append({
                    "proxy": pname, "ad_quantile": thr,
                    "delta_failrate_cliffminusout": float(np.mean(list(per_ds.values()))),
                    "n_datasets": len(per_ds),
                    "n_datasets_positive": pos,
                    "consistency": f"{pos}/{len(per_ds)}",
                })

    A = pd.DataFrame(all_rows)
    ds = pd.DataFrame(ds_rows)

    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 40)
    print("=" * 118)
    print("① 阈值 × 代理 的稳定性（每一行换一次 AD 界定）")
    print("=" * 118)
    show = ["proxy", "ad_quantile", "n_in_ad", "share_of_failures_in_ad",
            "fail_rate_in_ad", "fail_rate_out_ad", "cov_in_ad_cliff",
            "fail_in_ad_cliff", "cov_in_ad_noncliff", "n_cliff_out_ad",
            "in_ad_recall", "in_ad_recall_lift"]
    print(A[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 118)
    print("② 最刺眼的那条结论是否稳健：域内悬崖失败率 − 域外失败率（逐数据集方向一致性）")
    print("=" * 118)
    print(ds.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print("\n" + "=" * 118)
    print("③ 结论摘要")
    print("=" * 118)
    for r in A.itertuples():
        flag = "✓" if (r.n_cliff_out_ad == 0) else "✗"
        print(f"  {r.proxy:24s} AD>={r.ad_quantile:.0%}  "
              f"失效落在域内 {r.share_of_failures_in_ad:5.1f}%  "
              f"域内悬崖失败率 {r.fail_in_ad_cliff:.4f} vs 域外 {r.fail_rate_out_ad:.4f}  "
              f"域外悬崖数 {r.n_cliff_out_ad} {flag}  "
              f"域内召回 {r.in_ad_recall:.4f}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        A.to_csv(os.path.join(args.out, "AD_SENSITIVITY.csv"), index=False)
        ds.to_csv(os.path.join(args.out, "AD_SENSITIVITY_PERRUN.csv"), index=False)
        print(f"\n[saved] {args.out}/AD_SENSITIVITY*.csv")


if __name__ == "__main__":
    main()
