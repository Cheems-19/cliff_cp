#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 DOCX 里的表格重排为学术三线表（booktabs 风格）。

仅修改表格版式，不动图片（保护已修好的 A4 裁切修复）。
- 去掉所有竖线、去掉正文行间细线
- 顶线 / 底线 1.5pt，表头下中线 1pt
- 表头加粗 + 居中
- 数值列（多为数字、无中文）右对齐，近似小数点对齐
- 表格字号 9.5pt，宽度 100%

用法:
  python docx_booktabs.py <in.docx> <out.docx>
  python docx_booktabs.py <in.docx> --inspect      # 仅查看表结构
"""
import sys
import re
import copy
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# 含中文、CJK 标点（全角括号等）、日文假名区的都算“非数值”
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


def is_numeric_cell(text):
    """有数字、且不含任何中文/全角符号，即视为数值单元格。

    宽松处理：允许出现 – — （）等范围符号，只要不含 CJK 就判为数值。
    """
    t = text.strip()
    if not t:
        return False
    if CJK.search(t):
        return False
    return bool(re.search(r"\d", t))


def new_border(tag, val, sz, color="auto"):
    e = OxmlElement(f"w:{tag}")
    e.set(qn("w:val"), val)
    e.set(qn("w:sz"), str(sz))
    e.set(qn("w:space"), "0")
    e.set(qn("w:color"), color)
    return e


def clear_children(elem):
    for c in list(elem):
        elem.remove(c)


def inspect(docx_path):
    doc = Document(docx_path)
    print(f"表格数量: {len(doc.tables)}")
    for i, t in enumerate(doc.tables):
        rows = t.rows
        ncol = max(len(r.cells) for r in rows)
        # 数值列检测
        numeric_cols = []
        for j in range(ncol):
            body = [rows[r].cells[j].text for r in range(1, len(rows))]
            if body and sum(is_numeric_cell(c) for c in body) / len(body) >= 0.6:
                numeric_cols.append(j)
        head = " | ".join(c.text.strip().replace("\n", " ")[:18] for c in rows[0].cells)
        print(f"\n表#{i+1}: {len(rows)}行 x {ncol}列  数值列(右对齐)={numeric_cols}")
        print(f"   表头: {head}")


def apply_booktabs(docx_in, docx_out):
    doc = Document(docx_in)
    n_tables = len(doc.tables)
    print(f"处理 {n_tables} 个表格 ...")

    for ti, t in enumerate(doc.tables):
        tbl = t._tbl
        tblPr = tbl.tblPr

        # 1) 表格级边框：顶/底 1.5pt，其余 none
        borders = tblPr.find(qn("w:tblBorders"))
        if borders is None:
            borders = OxmlElement("w:tblBorders")
            tblPr.append(borders)
        clear_children(borders)
        borders.append(new_border("top", "single", 12))
        borders.append(new_border("bottom", "single", 12))
        for tag in ("left", "right", "insideH", "insideV"):
            borders.append(new_border(tag, "none", 0))

        # 2) 宽度 100%
        tblW = tblPr.find(qn("w:tblW"))
        if tblW is None:
            tblW = OxmlElement("w:tblW")
            tblPr.append(tblW)
        tblW.set(qn("w:w"), "5000")
        tblW.set(qn("w:type"), "pct")

        rows = t.rows
        ncol = max(len(r.cells) for r in rows)

        # 3) 数值列检测（基于正文行）
        numeric_cols = set()
        for j in range(ncol):
            body = [rows[r].cells[j].text for r in range(1, len(rows))]
            if body and sum(is_numeric_cell(c) for c in body) / len(body) >= 0.6:
                numeric_cols.add(j)

        # 4) 清空每个单元格自带边框（去掉行间细线）
        for r in rows:
            for c in r.cells:
                tc = c._tc
                tcPr = tc.tcPr
                if tcPr is not None:
                    b = tcPr.find(qn("w:tcBorders"))
                    if b is not None:
                        tcPr.remove(b)

        # 5) 表头：加粗 + 居中
        hdr = rows[0]
        for c in hdr.cells:
            for p in c.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.bold = True

        # 6) 正文单元格：数值列右对齐；统一字号 9.5pt
        for ri, r in enumerate(rows):
            for ci, c in enumerate(r.cells):
                for p in c.paragraphs:
                    if ri > 0 and ci in numeric_cols:
                        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                    for run in p.runs:
                        if run.font.size is None or run.font.size.pt != 9.5:
                            run.font.size = Pt(9.5)

        # 7) 表头下中线 1pt（mid-rule）
        row0 = hdr._tr
        trPr = row0.trPr
        if trPr is None:
            trPr = OxmlElement("w:trPr")
            row0.insert(0, trPr)
        rb = trPr.find(qn("w:tblBorders"))
        if rb is None:
            rb = OxmlElement("w:tblBorders")
            trPr.append(rb)
        clear_children(rb)
        rb.append(new_border("bottom", "single", 8))

        print(f"  表#{ti+1}: {len(rows)}行 x {ncol}列 | 数值列右对齐={sorted(numeric_cols)}")

    try:
        doc.save(docx_out)
        print(f"\n[已保存] {docx_out}")
    except PermissionError:
        alt = docx_in.rsplit(".docx", 1)[0] + "_booktabs.docx"
        doc.save(alt)
        print(f"\n[警告] 原文件被占用，已另存为: {alt}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    inp = sys.argv[1]
    if "--inspect" in sys.argv:
        inspect(inp)
    else:
        outp = sys.argv[2] if len(sys.argv) > 2 else inp
        apply_booktabs(inp, outp)
