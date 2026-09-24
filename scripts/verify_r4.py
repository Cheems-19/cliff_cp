# -*- coding: utf-8 -*-
"""修订轮 4：仅校验（apply 脚本第二次运行时编辑已全部落盘，此脚本补跑最终回读验证）。
只读，不写任何文件。
"""
import io
import re

ROOT = "."


def load(path):
    with io.open(f"{ROOT}/{path}", "r", encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


p = "论文全文_v1.md"
final = load(p)

# ---- 残留图号扫描 ----
stray = re.findall(r"图 \d+、\d", final)
assert not stray, f"stray figure refs: {stray}"

# ---- 16 项 probe ----
probes = [("初步发现", 0), ("d_pairgap", 0), ("R1 ——", 0), ("差距，§3.4", 0),
          ("富集 1.5–2.4 倍", 3), ("临界通过", 1), ("未在 OSF/AsPredicted 公开注册", 1),
          ("net_mag（无量纲比值）", 1), ("方向不一致", 1), ("按强度递增的六级偏移强度谱", 1),
          ("图 9、图 10", 1), ("图 11、图 12", 1), ("MIT 许可", 1),
          ("校准集真实标签", 1), ("统计单位约定", 1), ("Venn–ABERS", 1)]
for probe, expect in probes:
    got = final.count(probe)
    assert got == expect, f"probe {probe!r}: got {got}, expect {expect}"
    print(f"  probe OK x{expect}: {probe}")

# ---- 引用重编号后首现顺序 = 1..50 ----
TOK = r"\[(\d{1,2}(?:\s*[,，]\s*\d{1,2})*)\]"


def flatten(text):
    out = []
    for t in re.findall(TOK, text):
        out.extend(int(x) for x in re.split(r"[,，]", t))
    return out


tokens = flatten(final)
first = []
for v in tokens:
    if v not in first:
        first.append(v)
assert first == list(range(1, 51)), f"final citation order: {first}"
print(f"  citation first-appearance order OK (1..50, {len(tokens)} tokens)")

# ---- 正文重编号后关键句抽查 ----
for snip in ["三类[1]", "Norinder 等[3]", "Svensson 等[4]", "Mondrian 条件化**[30]",
             "Venn–ABERS 变体）[30]", "Adaptive Prediction Sets[49]", "Lee 等[48]", "Tursunbadalov 等[50]"]:
    assert snip in final, f"missing snippet: {snip}"
print("  renumbered citation snippets OK (7)")

# ---- 参考文献文件 ----
r2 = load("参考文献.md")
labels = [int(m.group(1)) for m in re.finditer(r"^\[(\d{1,2})\]", r2, re.M)]
assert labels == list(range(1, 51)), labels[:10]
assert r2.count("Tursunbadalov Muhammadjon, Tursunbadalov Mustafojon.") == 1
assert "[30] Vovk V, Gammerman A, Shafer G." in r2
assert r2.splitlines()[0].startswith("#") or "Algorithms" in r2
print("  references OK: labels 1..50, [30]=Vovk monograph, [50] full names")

# ---- 构建横幅 ----
b = load("scripts/build_paper_html.py")
assert "初稿 v5" in b and "初稿 v4" not in b
print("  build banner OK: v5")

print("VERIFY R4: ALL OK")
