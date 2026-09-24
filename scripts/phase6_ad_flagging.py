#!/usr/bin/env python
"""Phase 6：适用域（AD）过滤器抓不到悬崖失效——把它量化成可用的预警规则。

动机
----
到此为止的证据链是收敛的、但偏"否定"：共形侧的分区校准**在同等代价下修不好**悬崖覆盖。
一个纯否定的结论对论文不利。本轮把它推向"那该怎么办"。

可用的事实（Phase 1 的几何发现）：**活动悬崖位于化学空间最稠密处**
（最近邻相似度比非悬崖高 +0.125，300/300 一致）。
而所有适用域（AD）方法的判据都是"离训练集远 → 不可靠"。
两者方向相反 → **AD 过滤器在原理上就抓不到悬崖失效**。

本脚本量化三件事：
  1. 失效落在哪？按最近邻相似度分层，看覆盖率与失效占比
     → 关键数字：多大比例的失效发生在"AD 内部"
  2. 2×2 表：AD 内/外 × 悬崖/非悬崖 的覆盖率
     → 关键数字：AD 内的悬崖分子覆盖率是多少
  3. 固定预警预算下的召回率对比：AD 规则 vs 数据侧离散度规则 vs 两者合并
     → 关键数字：同等预警比例下，能多抓出多少"AD 抓不到的失效"

用法:
    python scripts/phase6_ad_flagging.py --dump results_ecfp_abl/dump --out results_ecfp_abl
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

NQ = 5  # 相似度分位数层数


def load(dump_dir):
    fs = sorted(glob.glob(os.path.join(dump_dir, "*.csv.gz")))
    if not fs:
        raise SystemExit(f"没找到 {dump_dir}/*.csv.gz")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def quantile_of(x, q):
    """返回 x 的组内分位（0..1），用于把分数变成可跨组比较的秩。"""
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return out
    order = stats.rankdata(x[ok], method="average")
    out[ok] = (order - 1) / (ok.sum() - 1)
    return out


def recall_at(scores, is_fail, frac):
    """组内按 scores 从高到低预警 frac 比例，返回失效召回率。"""
    n = len(scores)
    k = int(round(frac * n))
    if k < 1 or is_fail.sum() == 0:
        return np.nan, np.nan
    # 分数越高越该预警
    thr = np.sort(scores)[::-1][k - 1]
    flagged = scores >= thr
    rec = float(is_fail[flagged].sum() / is_fail.sum())
    return rec, float(flagged.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--flag-frac", type=float, default=0.20,
                    help="预警预算（按样本比例），默认 0.20。注意 argparse 会对 help 做 "
                         "百分号格式化，故此处不要写百分号")
    args = ap.parse_args()

    D = load(args.dump)
    print(f"载入 {len(D):,} 行 | {D['dataset'].nunique()} 数据集 | {D['seed'].nunique()} 种子")
    fail = ~D["covered_global"].to_numpy(bool)
    cliffB = D["cliffB"].to_numpy(bool)
    nnsim = D["d_nn_sim"].to_numpy(float)
    wstd = D["d_wstd"].to_numpy(float)
    base_rate = float(fail.mean())
    print(f"全局欠覆盖率 = {base_rate:.4f} | cliffB 占比 = {cliffB.mean():.4f}\n")

    # 关键坑：d_nn_sim 在"训练集里没有任何相似度 ≥ sim_floor 的邻居"时是 NaN
    # （见 dispersion.neighbor_label_stats：sel.size == 0 时 continue，未写 d_nn_sim）。
    # 这批分子恰恰是**最域外**的，直接丢掉会把 AD 的分析算偏。
    # 因此显式把 NaN 当作相似度 0（最疏），并单独报出数量。
    n_nan = int((~np.isfinite(nnsim)).sum())
    if n_nan:
        print(f"  注意：{n_nan} 条记录的最近邻相似度为 NaN（训练集内无匹配邻居）"
              f"→ 按相似度 0（最域外）处理")
    nnsim_ok = np.where(np.isfinite(nnsim), nnsim, 0.0)

    # 组内分位（跨组可比）
    grp = D.groupby(["dataset", "seed"], sort=False)
    idx_list = [ix for _, ix in grp.indices.items()]
    qsim = np.full(len(D), np.nan)
    qws = np.full(len(D), np.nan)
    for ix in idx_list:
        qsim[ix] = quantile_of(nnsim_ok[ix], None)
        qws[ix] = quantile_of(np.where(np.isfinite(wstd[ix]), wstd[ix], np.nan), None)
    qws_f = np.where(np.isfinite(qws), qws, 0.0)   # 无法算分散度 -> 不靠它预警

    # ---------------- 1) 失效按相似度分层 ----------------
    print("=" * 100)
    print("① 失效落在哪？按「最近邻相似度」分层（组内五分位，0=最疏，4=最稠）")
    print("=" * 100)
    rows = []
    for b in range(NQ):
        m = np.isfinite(qsim) & (np.floor(qsim * NQ).clip(0, NQ - 1) == b)
        if m.sum() == 0:
            continue
        rows.append({
            "sim_bin": b,
            "n": int(m.sum()),
            "share_of_all": m.mean() * 100,
            "coverage": float((~fail[m]).mean()),
            "fail_rate": float(fail[m].mean()),
            "share_of_failures": float(fail[m].sum() / fail.sum()) * 100,
            "cliffB_rate": float(cliffB[m].mean()) * 100,
        })
    S = pd.DataFrame(rows)
    print(S.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    in_ad = np.isfinite(qsim) & (qsim >= 0.20)   # 最稠的 80% = 「在 AD 内」
    print(f"\n  → AD 内（相似度 ≥ 20% 分位）承载了 **{fail[in_ad].sum()/fail.sum()*100:.1f}%** 的失效")

    # ---------------- 2) 2×2 表 ----------------
    print("\n" + "=" * 100)
    print("② 覆盖率 2×2：AD 内/外 × 悬崖/非悬崖  —— AD 保护不了悬崖")
    print("=" * 100)
    rows = []
    for ad_name, adm in (("AD 内", in_ad), ("AD 外", ~in_ad & np.isfinite(qsim))):
        for c_name, cm in (("悬崖(cliffB)", cliffB), ("非悬崖", ~cliffB)):
            m = adm & cm
            if m.sum() == 0:
                continue
            rows.append({
                "组": f"{ad_name} × {c_name}",
                "n": int(m.sum()),
                "share": m.mean() * 100,
                "coverage": float((~fail[m]).mean()),
                "fail_rate": float(fail[m].mean()),
                "share_of_failures": float(fail[m].sum() / fail.sum()) * 100,
            })
    T = pd.DataFrame(rows)
    print(T.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------- 3) 预警召回对比 ----------------
    print("\n" + "=" * 100)
    print("③ 固定预警预算下的失效召回（每组内按分数排名，再按失效数加权汇总）")
    print("=" * 100)
    fracs = [0.05, 0.10, 0.20, 0.30, 0.40]
    # 三个预警分数：值越大越应该被预警。
    # 关键：分位必须**组内**算，否则跨数据集不可比（不同数据集的 d_wstd 量纲差很多）。
    ad_score = 1.0 - qsim                      # 越疏越该预警（= 标准 AD 规则）
    disp_score = qws_f                         # 越分散越该预警
    both_score = np.maximum(ad_score, disp_score)  # 任一为高即预警
    rules = {
        "AD 规则（预警最疏的）": ad_score,
        "离散度规则（预警最分散的）": disp_score,
        "合并（两者都算预警）": both_score,
    }
    curves = []
    for name, sc in rules.items():
        sc = np.asarray(sc, float)
        for f in fracs:
            num = den = 0.0
            for ix in idx_list:
                r, _ = recall_at(sc[ix], fail[ix], f)
                if np.isfinite(r):
                    num += r * fail[ix].sum()
                    den += fail[ix].sum()
            curves.append({"rule": name, "flag_frac": f,
                           "failure_recall": num / den if den else np.nan})
    C = pd.DataFrame(curves)
    piv = C.pivot_table(index="flag_frac", columns="rule", values="failure_recall")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

    f0 = args.flag_frac
    row = piv.loc[f0] if f0 in piv.index else piv.iloc[-1]
    print(f"\n  在 {f0*100:.0f}% 预警预算下：")
    for k, v in row.items():
        print(f"    {k:26s} 失效召回 = {v:.4f}   （基准 = 随机预警 {f0:.2f}）")

    # ---------------- 4) 唯一的增量：AD 抓不到的那些 ----------------
    print("\n" + "=" * 100)
    print("④ 增量在哪：只看「AD 内部」的失效，离散度规则还能抓出多少")
    print("=" * 100)
    num = den = 0.0
    num_rand = 0.0
    for ix in idx_list:
        m = in_ad[ix]
        if m.sum() < 20:
            continue
        f_ = fail[ix][m]
        s_ = disp_score[ix][m]          # 用组内分位，且已把 NaN 归零
        if f_.sum() == 0:
            continue
        r, fr = recall_at(s_, f_, f0)
        if np.isfinite(r):
            num += r * f_.sum()
            den += f_.sum()
            num_rand += f0 * f_.sum()
    print(f"  AD 内部失效总数（分母）= {int(den):,}")
    print(f"  随机预警 {f0:.0%} 可抓到 = {num_rand/den:.4f}")
    print(f"  离散度规则预警 {f0:.0%} 可抓到 = {num/den:.4f}")
    print(f"  → 相对随机的提升 = {(num/den)/(f0):.2f}×")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        S.to_csv(os.path.join(args.out, "AD_BY_SIMBIN.csv"), index=False)
        T.to_csv(os.path.join(args.out, "AD_2x2.csv"), index=False)
        C.to_csv(os.path.join(args.out, "AD_RECALL_CURVES.csv"), index=False)
        print(f"\n[saved] {args.out}/AD_*.csv")


if __name__ == "__main__":
    main()
