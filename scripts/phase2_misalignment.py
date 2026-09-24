#!/usr/bin/env python
"""Phase 2：把「分区错配」从推理变成直接测量。

要回答的三个问题
----------------
Q-A  各个信号能"看见"活动悬崖吗？            AUC(信号 -> 是否 cliffB)
Q-B  各个信号能预测"会落在区间外"吗？          AUC(信号 -> 是否被欠覆盖)
     —— 两者之差就是错配的直接量化。
Q-C  错配是怎么造成损害的？                   按信号分箱的交叉表：
     每个分箱里 cliffB 占多少、该箱的分位数被调大还是调小、cliffB 在该箱的覆盖率如何变化。

第三个问题是关键：如果 cliffB 分子散落在分位数被**调小**的箱里，
它们就会在换用该分区时失去原本"搭便车"得到的余量，覆盖率随之下降。
这把"因为信号看不见悬崖、所以覆盖率被错配"从推理变成了可查的数字。

用法：
  python scripts/phase2_misalignment.py --dump results/phase1/dump --out results/phase2
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
from cliffcp import stats as S  # noqa: E402

SIGNALS = ["rf_std", "d_wstd", "d_std", "d_iqr", "d_nb", "d_nn_sim", "d_pair_gap"]
ORIENT = {  # +1: 越大越不可靠; -1: 越大越可靠（比较前先翻转）
    "rf_std": +1, "d_wstd": +1, "d_std": +1, "d_iqr": +1, "d_pair_gap": +1,
    "d_nb": -1, "d_nn_sim": -1,
}


def load(dump_dir):
    fs = sorted(glob.glob(os.path.join(dump_dir, "*_molecules.csv.gz")))
    if not fs:
        raise SystemExit(f"no dump files under {dump_dir}")
    parts = [pd.read_csv(f) for f in fs]
    d = pd.concat(parts, ignore_index=True)
    print(f"[data] {len(fs)} datasets, {len(d):,} molecule-seed records", flush=True)
    return d


def auc_table(d, target, label):
    """每个信号对 target 的判别 AUC（跨数据集先取均值）。"""
    rows = []
    for s in SIGNALS:
        per_ds = []
        for ds, g in d.groupby("dataset"):
            a = S.auc_detect(ORIENT[s] * g[s].to_numpy(float), g[target].to_numpy(bool))
            if np.isfinite(a):
                per_ds.append(a)
        if per_ds:
            rows.append(
                {
                    "target": label,
                    "signal": s,
                    "auc": float(np.mean(per_ds)),
                    "auc_sd": float(np.std(per_ds, ddof=1)) if len(per_ds) > 1 else np.nan,
                    "n_ds": len(per_ds),
                }
            )
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def within_group_auc(d, group_col):
    """在组内分别算 AUC：判断信号的整体判别力是否由某一组主导。"""
    rows = []
    for s in SIGNALS:
        for name, mask in (("cliffB", d[group_col]), ("noncliffB", ~d[group_col])):
            sub = d[mask]
            per_ds = []
            for ds, g in sub.groupby("dataset"):
                if g["covered_global"].nunique() < 2 or len(g) < 30:
                    continue
                a = S.auc_detect(
                    ORIENT[s] * g[s].to_numpy(float),
                    (~g["covered_global"]).to_numpy(bool),
                )
                if np.isfinite(a):
                    per_ds.append(a)
            if per_ds:
                rows.append(
                    {
                        "signal": s, "subgroup": name,
                        "auc": float(np.mean(per_ds)),
                        "n_ds": len(per_ds),
                        "n_mol": int(mask.sum()),
                    }
                )
    return pd.DataFrame(rows)


def cross_table(d, bin_col):
    """核心证据：按信号分箱的交叉表。

    每个箱给出：样本数、cliffB 占比、全局 CP 的分位数、分区方法的分位数、
    以及 cliffB / 非 cliffB 在两个方法下的覆盖率。
    """
    rows = []
    for b, g in d.groupby(bin_col):
        n_cb = int(g["cliffB"].sum())
        qg = float(np.nanmean(g["q_global"]))
        qr = float(np.nanmean(g["q_rfstd"]))
        qw = float(np.nanmean(g["q_wstd"]))
        row = {
            "bin": int(b),
            "n": int(len(g)),
            "n_cliffB": n_cb,
            "frac_cliffB": n_cb / len(g) if len(g) else np.nan,
            "q_global": qg,
            "q_rfstd": qr,
            "q_wstd": qw,
            "delta_q_rfstd": qr - qg,
            "delta_q_wstd": qw - qg,
        }
        for tag, mask in (("cliffB", g["cliffB"]), ("noncliffB", ~g["cliffB"])):
            sub = g[mask]
            for m in ("global", "rfstd", "wstd"):
                c = sub[f"covered_{m}"]
                row[f"cov_{tag}_{m}"] = float(c.mean()) if len(sub) else np.nan
                row[f"n_{tag}"] = int(len(sub))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="results/phase1/dump")
    ap.add_argument("--out", default="results/phase2")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(args.dump)

    # 只保留用得上、且信号非缺失的行
    d = d.dropna(subset=SIGNALS).reset_index(drop=True)
    print(
        f"[data] after dropna: {len(d):,} rows | "
        f"cliffB 占比 {d['cliffB'].mean():.2%} | cliffA 占比 {d['cliffA'].mean():.2%} | "
        f"全局欠覆盖率 {1 - d['covered_global'].mean():.2%}",
        flush=True,
    )

    # -------- Q-A vs Q-B：错配的直接量化
    a_sees = auc_table(d, "cliffB", "看见悬崖(cliffB)")
    # auc_table 的 target 传 covered_global，得到的是"预测被覆盖"的判别力；
    # 我们要的是"预测失效"，所以取 1 - auc（AUC 对类别翻转是对称的）。
    a_fail = auc_table(d, "covered_global", "预测失效")
    a_fail["auc"] = 1.0 - a_fail["auc"]
    a_sees.to_csv(out / "A_sees_cliff.csv", index=False)
    a_fail.to_csv(out / "B_predicts_failure.csv", index=False)

    print("\n" + "=" * 88)
    print("Q-A  信号能看见活动悬崖吗（AUC: 信号 -> 是否 cliffB；0.5 = 看不见）")
    print("=" * 88)
    print(a_sees[["signal", "auc", "auc_sd", "n_ds"]].to_string(index=False))

    print("\n" + "=" * 88)
    print("Q-B  信号能预测失效吗（AUC: 信号 -> 是否被欠覆盖）")
    print("=" * 88)
    print(a_fail[["signal", "auc", "auc_sd", "n_ds"]].to_string(index=False))

    merged = a_sees.merge(a_fail, on="signal", suffixes=("_sees_cliff", "_predicts_fail"))
    merged["gap_predict_minus_see"] = (
        merged["auc_predicts_fail"] - merged["auc_sees_cliff"]
    )
    merged = merged.sort_values("gap_predict_minus_see", ascending=False)
    print("\n" + "=" * 88)
    print("错配量：预测失效的能力 − 看见悬崖的能力（越大 = 越偏科，越容易在悬崖上翻车）")
    print("=" * 88)
    print(
        merged[
            ["signal", "auc_sees_cliff", "auc_predicts_fail", "gap_predict_minus_see"]
        ].to_string(index=False)
    )
    merged.to_csv(out / "C_misalignment.csv", index=False)

    # -------- Q-C：组内分解
    wg = within_group_auc(d, "cliffB")
    print("\n" + "=" * 88)
    print("组内判别力：信号在 cliffB 子集内 vs 非 cliffB 子集内对失效的 AUC")
    print("=" * 88)
    piv = wg.pivot_table(index="signal", columns="subgroup", values="auc")
    piv["n_mol_cliffB"] = wg[wg.subgroup == "cliffB"].set_index("signal")["n_mol"]
    print(piv.to_string())
    wg.to_csv(out / "D_within_group_auc.csv", index=False)

    # -------- Q-C：交叉表
    print("\n" + "=" * 88)
    print("交叉表（按 rf_std 分箱）：cliffB 分子落在哪些箱、该箱分位数被调大还是调小、覆盖率如何变")
    print("=" * 88)
    ct_rf = cross_table(d, "bin_rfstd")
    cols = ["bin", "n", "n_cliffB", "frac_cliffB", "q_global", "q_rfstd", "delta_q_rfstd",
            "cov_cliffB_global", "cov_cliffB_rfstd", "cov_noncliffB_global", "cov_noncliffB_rfstd"]
    print(ct_rf[cols].to_string(index=False))
    ct_rf.to_csv(out / "E_cross_rfstd.csv", index=False)

    print("\n" + "=" * 88)
    print("交叉表（按 d_wstd 分箱）：同一张表，看 wstd 分区为何不伤悬崖组")
    print("=" * 88)
    ct_ws = cross_table(d, "bin_wstd")
    cols2 = ["bin", "n", "n_cliffB", "frac_cliffB", "q_global", "q_wstd", "delta_q_wstd",
             "cov_cliffB_global", "cov_cliffB_wstd", "cov_noncliffB_global", "cov_noncliffB_wstd"]
    print(ct_ws[cols2].to_string(index=False))
    ct_ws.to_csv(out / "F_cross_wstd.csv", index=False)

    # -------- 结论：错配导致的损失分解
    print("\n" + "=" * 88)
    print("结论：cliffB 覆盖率变化的分解")
    print("=" * 88)
    cb = d[d["cliffB"]]
    for m in ("global", "rfstd", "wstd"):
        c = cb[f"covered_{m}"]
        ci = S.wilson_ci(int(c.sum()), len(c))
        print(
            f"  {m:8s} cliffB 覆盖率 = {c.mean():.4f}  "
            f"[{ci[0]:.4f}, {ci[1]:.4f}]  n={len(c):,}",
            flush=True,
        )
    print(
        "\n  cliffB 平均 rf_std = {:.4f}，非 cliffB = {:.4f}（差 {:.4f}）".format(
            cb["rf_std"].mean(), d[~d["cliffB"]]["rf_std"].mean(),
            cb["rf_std"].mean() - d[~d["cliffB"]]["rf_std"].mean(),
        ),
        flush=True,
    )
    print(
        "  cliffB 平均 d_wstd = {:.4f}，非 cliffB = {:.4f}（差 {:.4f}）".format(
            cb["d_wstd"].mean(), d[~d["cliffB"]]["d_wstd"].mean(),
            cb["d_wstd"].mean() - d[~d["cliffB"]]["d_wstd"].mean(),
        ),
        flush=True,
    )
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
