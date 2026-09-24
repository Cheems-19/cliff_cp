import sys, os

PLUGIN_SCRIPTS = r"C:/Users/Administrator/.workbuddy/plugins/cache/workbuddy-builtin/tencent-docx/5.5.6-wb.38337834.g5f969292.h4918b5ced607/skills/html-to-docx/scripts"
if PLUGIN_SCRIPTS not in sys.path:
    sys.path.insert(0, PLUGIN_SCRIPTS)

from html_to_docx import convert, ConvertOptions

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_academic import apply_academic

html_path = r"C:/Users/Administrator/Desktop/小论文/cliff_cp/figures/论文全文_v1.html"
out_path = r"C:/Users/Administrator/Desktop/小论文/cliff_cp/共形预测在活性悬崖上的覆盖失效_论文v5.docx"
base_dir = os.path.dirname(html_path)

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

print("html length:", len(html), "base_dir:", base_dir)

opts = ConvertOptions(
    page_size="A4", orientation="portrait",
    margin_top=2.54, margin_bottom=2.54, margin_left=3.17, margin_right=3.17,
    base_dir=base_dir, output_path=out_path,
)

res = convert(html, output_path=None, options=opts)
print("success:", res.success)

# 学术化后处理：图片安全（防裁切/去 alpha）+ 三线表
apply_academic(res.docx_path, out_path)
print("academic post-process done ->", out_path)
print("docx_path:", res.docx_path)
print("num_warnings:", len(res.warnings))
for w in res.warnings[:30]:
    print("  WARN:", w)
if not res.success:
    print("ERROR:", res.error)
else:
    print("OUTPUT EXISTS:", os.path.exists(res.docx_path), "size:", os.path.getsize(res.docx_path))
