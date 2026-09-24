#!/usr/bin/env python
"""把本地版与本远端版逐行 diff，找出那 2 字节的差异到底在哪。"""
import difflib
import os
import sys
from pathlib import Path

# 项目根：默认当前目录，或用环境变量 CLIFFCP_ROOT 指定
ROOT = Path(os.environ.get("CLIFFCP_ROOT", "."))
a = (ROOT / "scripts/phase1_signal_local.py").read_text(encoding="utf-8").splitlines()
b = (ROOT / "scripts/phase1_signal.py").read_text(encoding="utf-8").splitlines()

d = list(difflib.unified_diff(a, b, fromfile="local", tofile="remote", lineterm="", n=1))
if not d:
    print("两份文件逐行一致（差异只在不可见字符或行尾）")
else:
    print(f"发现 {len(d)} 行 diff：")
    for line in d[:80]:
        print(line)

# 再看不可见字符层面
ba = (ROOT / "scripts/phase1_signal_local.py").read_bytes()
bb = (ROOT / "scripts/phase1_signal.py").read_bytes()
print(f"\nlocal bytes={len(ba)}  remote bytes={len(bb)}  delta={len(bb)-len(ba)}")
for name, buf in (("local", ba), ("remote", bb)):
    print(f"  {name}: CR={buf.count(bytes([13]))} LF={buf.count(bytes([10]))} "
          f"TAB={buf.count(bytes([9]))} trail_nl={buf.endswith(bytes([10]))}")
