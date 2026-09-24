#!/usr/bin/env python
"""把阈值敏感性的各配置结果汇成一张长表。

用法: python scripts/collect_sensitivity.py --root results [--glob 'sens_*']
输出: 打印一张"配置 × 分区变量"的长表，并存 root/SENSITIVITY.csv
"""
import argparse
import glob
import os
import re

import pandas as pd

TAG_RE = re.compile(r"(?:sens|alpha)_([a-z0-9]+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--glob", default="sens_*")
    ap.add_argument("--methods", default="",
                    help="只保留方法名前缀匹配的行，如 'mondrian_'。"
                         "用途：当扫描中途新增了对比方法（如 cqr）时，"
                         "用过滤保证早期配置与晚期配置在同一行集合上比较。")
    args = ap.parse_args()

    frames = []
    for d in sorted(glob.glob(os.path.join(args.root, args.glob))):
        f = os.path.join(d, "PHASE1_ABLATION.csv")
        if not os.path.isfile(f):
            continue
        tag = TAG_RE.search(os.path.basename(d).rstrip("/"))
        tag = tag.group(1) if tag else os.path.basename(d)
        t = pd.read_csv(f)
        t.insert(0, "config", tag)
        frames.append(t)
    if not frames:
        print("没有找到任何 PHASE1_ABLATION.csv —— 检查 --root/--glob")
        return

    A = pd.concat(frames, ignore_index=True)
    if args.methods:
        pre = [p.strip() for p in args.methods.split(",") if p.strip()]
        keep = A["method"].astype(str).str.startswith(tuple(pre))
        dropped = sorted(set(A.loc[~keep, "method"].astype(str)))
        A = A[keep]
        if dropped:
            print(f"[filter] 已排除方法: {', '.join(dropped)}\n")
    cols = [c for c in [
        "config", "method", "delta_mean", "wilcoxon_p", "n_ds_delta_pos", "n_ds",
        "meta_delta", "meta_p", "meta_I2", "worst_group_delta", "width_delta",
    ] if c in A.columns]
    A = A[cols]

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)

    print("=" * 110)
    print("阈值敏感性：各配置下，各分区变量对【悬崖组覆盖率】的影响（delta_mean）")
    print("=" * 110)
    piv = A.pivot_table(index="method", columns="config", values="delta_mean")
    print(piv.to_string(float_format=lambda v: f"{v:+.4f}"))

    print("\n" + "=" * 110)
    print("方向一致性（多少个数据集上为正）")
    print("=" * 110)
    piv2 = A.pivot_table(index="method", columns="config", values="n_ds_delta_pos")
    nds = A.pivot_table(index="method", columns="config", values="n_ds")
    comb = piv2.astype("Int64").astype(str) + "/" + nds.astype("Int64").astype(str)
    print(comb.to_string())

    print("\n" + "=" * 110)
    print("宽度代价（正值 = 区间变宽）")
    print("=" * 110)
    piv3 = A.pivot_table(index="method", columns="config", values="width_delta")
    print(piv3.to_string(float_format=lambda v: f"{v:+.4f}"))

    out = os.path.join(args.root, "SENSITIVITY.csv")
    A.to_csv(out, index=False)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
