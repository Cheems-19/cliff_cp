#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DOCX 学术化后处理（可复现流水线用）。

两步：
  1) 图片安全：把超过 A4 可用宽（14.66cm）的图等比缩到页面内（防裁切）；
     把 RGBA 图压成白底 RGB（去 SMask 膨胀）。已合规的图幂等不动。
  2) 三线表：调用 docx_booktabs 的逻辑（去竖线/行间线、顶底 1.5pt、
     表头下 1pt、表头加粗居中、数值列右对齐、9.5pt）。

用法（在 convert_paper.py 末尾调用）：
  from docx_academic import apply_academic
  apply_academic(res.docx_path, out_path)
也可单独跑：
  python docx_academic.py <in.docx> [out.docx]
"""
import io
import os
import sys

from docx import Document
from docx.oxml.ns import qn

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

# A4 可用宽 = 21cm - 左右边距(3.17*2) = 14.66cm，转 EMU（1cm = 360000 EMU）
PAGE_W = int((21.0 - 6.34) * 360000)

# 复用表格逻辑：docx_booktabs 与本文件同目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_booktabs import apply_booktabs  # noqa: E402


def fix_images(doc):
    body = doc.element.body
    drawings = body.findall(".//" + qn("wp:inline")) + body.findall(
        ".//" + qn("wp:anchor")
    )
    scaled = 0
    for d in drawings:
        we = d.find(qn("wp:extent"))
        if we is None:
            continue
        try:
            cx = int(we.get("cx"))
            cy = int(we.get("cy"))
        except (TypeError, ValueError):
            continue
        if cx > 0 and cx > PAGE_W:
            s = PAGE_W / cx
            ncx, ncy = PAGE_W, int(round(cy * s))
            we.set("cx", str(ncx))
            we.set("cy", str(ncy))
            ae = d.find(".//" + qn("a:ext"))
            if ae is not None:
                ae.set("cx", str(ncx))
                ae.set("cy", str(ncy))
            scaled += 1

    flat = 0
    if HAVE_PIL:
        for rel in list(doc.part.rels.values()):
            if "image" in rel.reltype:
                part = rel.target_part
                try:
                    img = Image.open(io.BytesIO(part.blob))
                    if img.mode == "RGBA":
                        bg = Image.new("RGB", img.size, (255, 255, 255))
                        bg.paste(img, img)
                        buf = io.BytesIO()
                        bg.save(buf, "PNG")
                        part.blob = buf.getvalue()
                        flat += 1
                except Exception:
                    pass
    return scaled, flat


def apply_academic(docx_in, docx_out):
    doc = Document(docx_in)
    scaled, flat = fix_images(doc)
    print(f"[images] 超宽图缩放={scaled}  RGBA压平={flat}")

    tmp = docx_in + ".__academic_tmp__.docx"
    doc.save(tmp)
    try:
        apply_booktabs(tmp, docx_out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return scaled, flat


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    inp = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else inp
    apply_academic(inp, outp)
