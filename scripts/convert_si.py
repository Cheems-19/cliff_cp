# -*- coding: utf-8 -*-
"""SI_v1.md -> html -> SI_v1.docx（复用官方 html_to_docx 包）"""
import io, os, sys

C = r"C:/Users/Administrator/Desktop/小论文/cliff_cp"
PLUGIN = (r"C:/Users/Administrator/.workbuddy/plugins/cache/workbuddy-builtin/tencent-docx/"
          r"5.5.6-wb.38337834.g5f969292.h4918b5ced607/skills/html-to-docx/scripts")
if PLUGIN not in sys.path:
    sys.path.insert(0, PLUGIN)

import markdown as mdlib
from html_to_docx import convert, ConvertOptions

md_text = io.open(os.path.join(C, "SI_v1.md"), encoding="utf-8").read()
body = mdlib.markdown(md_text, extensions=["tables", "sane_lists", "fenced_code"])

CSS = """
<style>
body { font-family: "Times New Roman", "SimSun", serif; font-size: 11pt; line-height: 1.5; }
h1 { font-size: 16pt; } h2 { font-size: 14pt; margin-top: 1.2em; } h3 { font-size: 12pt; }
h4, h5, h6 { font-size: 11pt; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 0.6em 0; }
th, td { border: 1px solid #999; padding: 3px 5px; text-align: left; }
th { background: #f0f0f0; }
code { font-family: Consolas, monospace; font-size: 9.5pt; }
pre { background: #f6f6f6; padding: 6px; font-size: 9pt; }
blockquote { border-left: 3px solid #bbb; margin-left: 0; padding-left: 10px; color: #333; }
</style>
"""
html = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Supporting Information</title>" + CSS + "</head><body>" + body + "</body></html>")

html_path = os.path.join(C, "SI_v1.html")
io.open(html_path, "w", encoding="utf-8").write(html)
out_path = os.path.join(C, "SI_v1.docx")

opts = ConvertOptions(
    page_size="A4", orientation="portrait",
    margin_top=2.54, margin_bottom=2.54, margin_left=2.54, margin_right=2.54,
    base_dir=os.path.dirname(html_path), output_path=out_path,
)
res = convert(html, output_path=None, options=opts)
print("success:", res.success, "| warnings:", len(res.warnings), "| out:", res.docx_path)
for x in res.warnings[:20]:
    print("  WARN:", x)
if not res.success:
    print("ERROR:", res.error)
else:
    print("exists:", os.path.exists(res.docx_path), "size:", os.path.getsize(res.docx_path))
