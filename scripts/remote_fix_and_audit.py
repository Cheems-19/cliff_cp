#!/usr/bin/env python
"""修掉远端 phase1_signal.py 的拼写错误，并对所有上传过的文件做 md5 对账。

背景：scp 送达的 phase1_signal.py 与本地不一致（本地 29857 B，远端 29854 B），
差异是 3 处 `RandomForestRegresso`（少了结尾的 r）。py_compile 只查语法，查不出来，
于是 α 扫描 5 个配置全部在 ImportError 上静默失败。

教训：**上传之后必须做"真实验证"——import 一次并实例化关键对象**，
而不是只跑 py_compile。本脚本就是这个验证器。

用法: python scripts/remote_fix_and_audit.py [--fix]
"""
import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLIFFCP_ROOT", "."))

# 关键文件：脚本名 -> 是否需要能被 import
KEY_FILES = [
    "scripts/phase1_signal.py",
    "scripts/phase1_analyze.py",
    "scripts/collect_sensitivity.py",
    "scripts/coverage_matched.py",
    "scripts/meta_analysis.py",
    "src/cliffcp/__init__.py",
    "src/cliffcp/data.py",
    "src/cliffcp/features.py",
    "src/cliffcp/cliffs.py",
    "src/cliffcp/conformal.py",
    "src/cliffcp/stats.py",
    "src/cliffcp/dispersion.py",
]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()

    target = ROOT / "scripts/phase1_signal.py"
    s = target.read_text(encoding="utf-8")
    # 已观测到的两类损坏：类名结尾的 "r" 被吃掉。
    # \b 很关键：'Regresso' 是 'Regressor' 的前缀，不能用普通替换，
    # 否则会把正确的 'Regressor' 改成 'Regressorr'。
    patterns = [
        (r"RandomForestRegresso\b", "RandomForestRegressor"),
        (r"HistGradientBoostingRegresso\b", "HistGradientBoostingRegressor"),
    ]
    n_total = 0
    for pat, rep in patterns:
        hits = re.findall(pat, s)
        if hits:
            print(f"[audit] {target.name}: 发现 {len(hits)} 处 '{pat}'")
            n_total += len(hits)
            s = re.sub(pat, rep, s)
    if n_total and args.fix:
        target.write_text(s, encoding="utf-8")
        print(f"[fix] 已写入修正版（{len(s)} 字节）")
    elif n_total:
        print("[fix] 未加 --fix，只报告不修改")
    else:
        print("[audit] 未发现已知拼写损坏")

    print("\n[audit] 各关键文件 md5 与大小：")
    for rel in KEY_FILES:
        p = ROOT / rel
        if not p.exists():
            print(f"  MISSING  {rel}")
            continue
        print(f"  {md5(p)}  {p.stat().st_size:>8}  {rel}")

    print("\n[verify] 真实 import 与实例化（这才是有效的验证）：")
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib

    ok = True
    for modname in ["cliffcp.conformal", "cliffcp.stats", "cliffcp.data",
                    "cliffcp.features", "cliffcp.cliffs", "cliffcp.dispersion",
                    "phase1_signal", "phase1_analyze",
                    "collect_sensitivity", "coverage_matched"]:
        try:
            m = importlib.import_module(modname)
            extra = ""
            if modname == "phase1_signal":
                extra = f"  make_model('rf')={type(m.make_model('rf', 0)).__name__}"
            print(f"  OK   {modname}{extra}")
        except Exception as e:
            ok = False
            print(f"  FAIL {modname}: {type(e).__name__}: {e}")
    if not ok:
        print("\n!! 存在导入失败 —— 不要基于这个状态启动任何批跑")
        return 1
    print("\n[verify] 全部通过，可以启动批跑")
    return 0


if __name__ == "__main__":
    sys.exit(main())
