# -*- coding: utf-8 -*-
"""修订轮 4：DOCX 交付物 QA（反转义后 probe 检查）。只读。"""
import re
import zipfile
from xml.sax.saxutils import unescape

DOCX = "共形预测在活性悬崖上的覆盖失效_论文v5.docx"

with zipfile.ZipFile(DOCX) as z:
    xml = z.read("word/document.xml").decode("utf-8")

# 段落切分（保留段落边界避免跨段误拼接），再剥标签
xml = re.sub(r"</w:p>", "\n", xml)
text = unescape(re.sub(r"<[^>]+>", "", xml))

fails = []

# ---- 必须为 0 的残留 ----
for probe, expect in [("初步发现", 0), ("d_pairgap", 0), ("逐行 +0.618", 0),
                      ("R1 ——", 0), ("差距，§3.4", 0)]:
    got = text.count(probe)
    ok = got == expect
    if not ok:
        fails.append((probe, got, expect))
    print(f"  {'OK ' if ok else 'FAIL'} x{expect}: {probe} (got {got})")

# ---- 新内容必须存在 ----
new_probes = [("富集 1.5–2.4 倍", 3), ("临界通过", 1),
              ("未在 OSF/AsPredicted 公开注册", 1),
              ("方向不一致", 1), ("图 9、图 10", 1), ("图 11、图 12", 1),
              ("MIT 许可", 1), ("按校准集真实标签", 1), ("统计单位约定", 1),
              ("Venn–ABERS", 1), ("初稿 v5", 1),
              ("三类[1]", 1), ("Norinder 等[3]", 1), ("Svensson 等[4]", 1),
              ("Mondrian 条件化[30]", 1), ("Adaptive Prediction Sets[49]", 1),
              ("Lee 等[48]", 1), ("Tursunbadalov 等[50]", 1),
              ("与悬崖轴正交", 1), ("悬崖组 0.439 vs 0.279", 1),
              ("最疏十分位 18.1 → 最密十分位 17.4 pp", 1)]
for probe, expect in new_probes:
    got = text.count(probe)
    ok = got >= expect
    if not ok:
        fails.append((probe, got, expect))
    print(f"  {'OK ' if ok else 'FAIL'} >= {expect}: {probe} (got {got})")

# ---- 参考文献块 ----
assert "Tursunbadalov Muhammadjon, Tursunbadalov Mustafojon." in text, "[50] full names missing"
assert re.search(r"\[30\]\s*Vovk V, Gammerman A, Shafer G\.", text), "[30] Vovk missing"
assert re.search(r"\[1\]\s*Norinder", text) or "Algorithms" in text, "[1] first entry"
print("  OK  references: [50] full names / [30] Vovk / [1] Norinder(Algorithms)")

# ---- [1] 首条应为 Norinder（MAPIE 原文）确认 ----
m = re.search(r"\[1\]([^\[]{20,120})", text)
print(f"  [1] entry head: {m.group(1)[:60].strip() if m else 'N/A'}")

# ---- 渲染层空格容忍的检查（DOCX 转换器在拉丁/CJK 边界插空格）----
m = re.search(r"net_mag\s*（无量纲比值）", text)
print(f"  {'OK ' if m else 'FAIL'} regex: net_mag（无量纲比值） (space-tolerant)")
if not m:
    fails.append(("net_mag（无量纲比值）", 0, 1))

print("DOCX QA:", "FAIL x%d" % len(fails) if fails else "ALL OK")
if fails:
    raise SystemExit(1)
