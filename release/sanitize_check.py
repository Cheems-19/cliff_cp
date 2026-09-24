#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""sanitize_check.py —— 发布前敏感信息自查（默认只报告，不改动）

用法
----
    python release/sanitize_check.py                     # 扫描当前项目（默认根目录为脚本上级）
    python release/sanitize_check.py --root /path/to/proj
    python release/sanitize_check.py --fix                # 就地替换为占位符（先备份为 .bak）
    python release/sanitize_check.py --fix --yes          # 跳过二次确认

为什么需要它
------------
论文配套仓库常见的"发布事故"是：服务器 IP / 用户名 / 绝对路径 散落在
运维脚本、日志、报告里，`git push` 后被搜索引擎永久索引。
本脚本把这类字符串集中扫出来，让 30 秒的人工复核替代逐文件 grep。

⚠️ `--fix` 之后**务必人工复核**：自动替换可能改到结果文件里本来就有意义的字符串。
"""
from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import sys

# (正则, 替换为, 说明)
RULES = [
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d{2,5})?\b"),
     "{SERVER}",
     "IPv4 地址（含端口）"),
    (re.compile(r"\bssh\s+-p\s+\d+\s+[\w.\-]+@\{SERVER\}"),
     "ssh {SERVER}",
     "ssh 命令中的主机凭据"),
    (re.compile(r"/mnt/[A-Za-z][\w.\-]*/"),
     "/path/to/",
     "服务器绝对路径（含用户名）"),
    (re.compile(r"\b(?:password|passwd|token|secret|api[_-]?key)\s*[=:]\s*\S+", re.I),
     "{REDACTED}",
     "疑似凭据赋值"),
    (re.compile(r"\b(?:AKIA|ghp_|github_pat_|sk-)[A-Za-z0-9_\-]{12,}"),
     "{REDACTED}",
     "疑似 API key / 访问令牌"),
]

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
             "wheels_cu121", "wheels_cu124", ".ipynb_checkpoints"}
TEXT_EXT = {".py", ".md", ".txt", ".sh", ".json", ".yaml", ".yml", ".cff",
            ".csv", ".tsv", ".log", ".cfg", ".toml", ".ini", ".html", ".xml"}
MAX_BYTES = 8 * 1024 * 1024   # 跳过 >8MB 的文件（二进制缓存等）


def iter_files(root: str):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            p = os.path.join(dp, f)
            ext = os.path.splitext(f)[1].lower()
            if ext not in TEXT_EXT:
                continue
            try:
                if os.path.getsize(p) > MAX_BYTES:
                    continue
            except OSError:
                continue
            yield p


def scan(root: str):
    hits = []          # (path, lineno, rule_desc, matched_text, line)
    for p in iter_files(root):
        try:
            with io.open(p, encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            for rx, _, desc in RULES:
                m = rx.search(line)
                if m:
                    hits.append((p, i, desc, m.group(0), line.rstrip()[:160]))
    return hits


def fix(root: str, hits, assume_yes: bool):
    if not hits:
        print("没有需要修改的内容。")
        return 0
    if not assume_yes:
        print(f"将就地修改 {len({h[0] for h in hits})} 个文件，并把原文件备份为 *.bak。")
        ans = input("继续？输入 yes 确认：").strip().lower()
        if ans != "yes":
            print("已取消。")
            return 1
    changed = 0
    for p in sorted({h[0] for h in hits}):
        try:
            with io.open(p, encoding="utf-8", errors="ignore") as fh:
                txt = fh.read()
        except OSError:
            continue
        new = txt
        for rx, rep, _ in RULES:
            new = rx.sub(rep, new)
        if new != txt:
            if not os.path.exists(p + ".bak"):
                shutil.copy2(p, p + ".bak")
            with io.open(p, "w", encoding="utf-8") as fh:
                fh.write(new)
            changed += 1
            print("  [fixed]", os.path.relpath(p, root))
    print(f"\n共修改 {changed} 个文件（原件保留为 *.bak）。请务必人工复核！")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="发布前敏感信息自查")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="项目根目录（默认：本脚本所在目录的上一级）")
    ap.add_argument("--fix", action="store_true", help="就地替换为占位符（先备份 .bak）")
    ap.add_argument("--yes", action="store_true", help="配合 --fix，跳过二次确认")
    args = ap.parse_args(argv)

    root = args.root
    if not os.path.isdir(root):
        print("目录不存在：", root)
        return 2

    print(f"扫描根目录：{root}\n")
    hits = scan(root)
    if not hits:
        print("✅ 未发现 IPv4 / 服务器路径 / 疑似凭据。可以发布。")
        print("   （提示：仍建议人工确认没有业务敏感的其它标识。）")
        return 0

    by_rule = {}
    for p, ln, desc, txt, _ in hits:
        by_rule.setdefault(desc, []).append((p, ln, txt))

    print(f"⚠️  发现 {len(hits)} 处疑似敏感字符串：\n")
    for desc, items in sorted(by_rule.items(), key=lambda x: -len(x[1])):
        files = sorted({os.path.relpath(p, root) for p, _, _ in items})
        print(f"── {desc}：{len(items)} 处，涉及 {len(files)} 个文件")
        for f in files[:8]:
            print(f"     - {f}")
        if len(files) > 8:
            print(f"     … 另 {len(files) - 8} 个文件")
        for p, ln, txt in items[:3]:
            print(f"       例：{os.path.relpath(p, root)}:{ln}  {txt[:100]}")
        print()

    if args.fix:
        return fix(root, hits, args.yes)

    print("如需批量替换为占位符：python release/sanitize_check.py --fix")
    print("（或按 release/.gitignore 直接排除 remote/ 与日志，再手工处理 README。）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
