#!/usr/bin/env python
"""诊断：coverage_matched 的 glob 到底匹配到了什么。"""
import glob
import os

os.chdir(os.environ.get("CLIFFCP_ROOT", "."))
print("cwd =", os.getcwd())
for g in ["alpha_*", "cqr_*"]:
    hits = sorted(glob.glob(os.path.join("results", g, "METHODS.csv")))
    print(f"\nglob {g!r} -> {len(hits)} 个")
    for h in hits[:8]:
        print("  ", h)
    missing = [d for d in sorted(glob.glob(os.path.join("results", g)))
               if os.path.isdir(d) and not os.path.exists(os.path.join(d, "METHODS.csv"))]
    if missing:
        print("  缺 METHODS.csv 的目录:", missing)
