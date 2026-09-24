#!/usr/bin/env python
"""把 论文全文_v1.md 转成学术版式的 HTML（供 HTML→DOCX 转换）。

版式：学术论文风——宋体正文、黑体标题、两端对齐、表格三线感、图表路径占位。
"""
import io
import os

import markdown

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

md_text = io.open("论文全文_v1.md", encoding="utf-8").read()

# 参考文献：单独维护（参考文献.md），构建时并入正文；忽略其内部附录（## 附：…）
try:
    ref_text = io.open("参考文献.md", encoding="utf-8").read()
    cut = ref_text.find("\n## 附：")
    if cut > 0:
        ref_text = ref_text[:cut]
    # 去掉文件标题行（# 参考文献）与前言引用块，只保留编号条目
    lines = [ln for ln in ref_text.splitlines()
             if not ln.startswith("# ") and not ln.startswith(">")]
    md_text = md_text.rstrip() + "\n\n---\n\n## 参考文献\n\n" + "\n".join(lines).strip() + "\n"
except FileNotFoundError:
    pass

CSS = """
body { font-family: "SimSun","宋体",serif; font-size: 11pt; line-height: 1.7;
       color: #1a1a1a; margin: 2.2cm 2.4cm; text-align: justify; }
h1 { font-family: "SimHei","黑体",sans-serif; font-size: 17pt; text-align: center;
     margin: 0 0 0.4em 0; }
h2 { font-family: "SimHei","黑体",sans-serif; font-size: 14pt; margin: 1.6em 0 0.6em;
     border-bottom: 1.5pt solid #333; padding-bottom: 3px; }
h3 { font-family: "SimHei","黑体",sans-serif; font-size: 12pt; margin: 1.2em 0 0.4em; }
h4 { font-family: "SimHei","黑体",sans-serif; font-size: 11pt; margin: 1em 0 0.3em; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 0.8em 0; }
th { border-top: 1.2pt solid #000; border-bottom: 0.8pt solid #000;
     font-family: "SimHei",sans-serif; padding: 4px 6px; text-align: left; }
td { padding: 3px 6px; border-bottom: 0.4pt solid #ccc; }
tr:last-child td { border-bottom: 1.2pt solid #000; }
code { font-family: Consolas,monospace; font-size: 9.5pt; background: #f4f4f4;
       padding: 0 2px; }
blockquote { border-left: 3pt solid #888; margin: 0.8em 0; padding: 0.2em 1em;
             color: #333; background: #f7f7f7; }
em { font-style: normal; font-weight: bold; }
.meta { color: #555; font-size: 10pt; text-align: center; margin-bottom: 1.5em; }
"""

body = markdown.markdown(md_text, extensions=["tables", "sane_lists"])

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>共形预测的覆盖保证在活性悬崖上如何失效</title>
<style>{CSS}</style></head>
<body>
<div class="meta">初稿 v5 · 2026-09-24 · 图 1–15 按出现顺序重编号 · 表 4 / 表 5 已回填 · 附口径速查与符号表（§2.9）</div>
{body}
</body></html>"""

out = "figures/论文全文_v1.html"
io.open(out, "w", encoding="utf-8").write(html)
print(f"[saved] {out}  ({len(html)} 字符)")
