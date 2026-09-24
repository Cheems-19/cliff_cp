# -*- coding: utf-8 -*-
"""把打印预览 PDF 里的 15 张图替换回 600dpi 原图（RGBA→白底RGB，防SMask膨胀）。

按 论文全文_v1.md 中图片出现顺序逐一对应替换；只换图像数据，不动版面。
"""
import io
import os
import re
import shutil

import fitz  # pymupdf
from PIL import Image

ROOT = r"C:/Users/Administrator/Desktop/小论文/cliff_cp"
FIG_DIR = os.path.join(ROOT, "figures")
MD = os.path.join(ROOT, "论文全文_v1.md")
PDF = os.path.join(ROOT, "共形预测在活性悬崖上的覆盖失效_论文v5_打印预览.pdf")
TMP_DIR = os.path.join(ROOT, "_session_tmp")
os.makedirs(TMP_DIR, exist_ok=True)
TMP_PDF = os.path.join(TMP_DIR, "v5_preview_hires.pdf")

# 1) 按出现顺序取图源
md = io.open(MD, encoding="utf-8").read()
figs = re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", md)
assert len(figs) == 15, f"预期 15 张图，实际 {len(figs)}"

# 2) 预转换：RGBA→白底 RGB，存临时 PNG
rgb_files = []
for i, f in enumerate(figs):
    img = Image.open(os.path.join(FIG_DIR, f))
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, img)
        img = bg
    p = os.path.join(TMP_DIR, f"_hires_{i:02d}.png")
    img.save(p, "PNG")
    rgb_files.append(p)

# 3) 打开 PDF，按文档顺序收集图片 xref
doc = fitz.open(PDF)
xrefs = []
for page in doc:
    for info in page.get_image_info(xrefs=True):
        xrefs.append(info["xref"])
xrefs = list(dict.fromkeys(xrefs))  # 去重保序
print(f"PDF 内图片 xref 数: {len(xrefs)}")
assert len(xrefs) == 15, "PDF 图片数与论文图数不一致，需人工核对"

# 4) 逐张替换
for xref, path in zip(xrefs, rgb_files):
    page = None
    for p in doc:
        if any(i["xref"] == xref for i in p.get_image_info(xrefs=True)):
            page = p
            break
    page.replace_image(xref, filename=path)
print("15 张图已替换为 600dpi 原图")

doc.save(TMP_PDF, garbage=3, deflate=True)
doc.close()
print(f"临时 PDF: {os.path.getsize(TMP_PDF)/1e6:.1f} MB")

# 5) 原子替换
try:
    shutil.move(TMP_PDF, PDF)
    print(f"replaced: {PDF} ({os.path.getsize(PDF)/1e6:.1f} MB)")
except PermissionError:
    print(f"[警告] 目标被占用，高分辨率版本保留在: {TMP_PDF}")

# 6) 清理临时 PNG
for p in rgb_files:
    os.remove(p)
print("done")
