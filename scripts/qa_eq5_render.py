# -*- coding: utf-8 -*-
"""把 v5 论文 DOCX 用 Word 排版导出 PDF，再渲染关键页为 PNG 供目检。"""
import os

import win32com.client as w32

ROOT = r"C:\Users\Administrator\Desktop\小论文\cliff_cp"
DOCX = os.path.join(ROOT, "共形预测在活性悬崖上的覆盖失效_论文v3.docx")
PDF = os.path.join(ROOT, "_qa_eq5.pdf")

app = w32.Dispatch("Word.Application")
app.Visible = False
app.DisplayAlerts = 0
try:
    doc = app.Documents.Open(DOCX, ReadOnly=True, AddToRecentFiles=False)
    doc.ExportAsFixedFormat(PDF, 17)  # 17 = wdExportFormatPDF
    pages = doc.ComputeStatistics(2)  # 2 = wdStatisticPages
    print("PDF exported, pages =", pages)
    doc.Close(False)
finally:
    app.Quit()

import sys

sys.path.insert(0, os.path.join(ROOT, ".tmp_pkgs", "venv", "Lib", "site-packages"))
import fitz  # PyMuPDF

pdf = fitz.open(PDF)
print("pdf pages:", pdf.page_count)
for pno in (7, 8, 9):  # 0-based: 第8、9、10页
    if pno >= pdf.page_count:
        break
    page = pdf[pno]
    pix = page.get_pixmap(dpi=130)
    out = os.path.join(ROOT, f"_qa_eq5_p{pno + 1}.png")
    pix.save(out)
    print("saved", out)
    # 同时抽取该页文本，报告式(5)所在页
    txt = page.get_text()
    if "net_mag(G" in txt:
        print(f"  -> formula (5) found on page {pno + 1}")
