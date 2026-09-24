# -*- coding: utf-8 -*-
"""用 Word COM 导出打印预览 PDF（先导出到临时文件再原子替换，规避占用）。"""
import os
import shutil
import win32com.client

SRC = r"C:/Users/Administrator/Desktop/小论文/cliff_cp/共形预测在活性悬崖上的覆盖失效_论文v5.docx"
OUT = r"C:/Users/Administrator/Desktop/小论文/cliff_cp/共形预测在活性悬崖上的覆盖失效_论文v5_打印预览.pdf"
TMP_DIR = r"C:/Users/Administrator/Desktop/小论文/cliff_cp/_session_tmp"
os.makedirs(TMP_DIR, exist_ok=True)
TMP = os.path.join(TMP_DIR, "v5_preview.pdf")

word = win32com.client.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
try:
    doc = word.Documents.Open(SRC)  # 不用 ReadOnly，避免导出被拦
    doc.ExportAsFixedFormat(TMP, 17)  # 17 = wdExportFormatPDF
    n_pages = doc.ComputeStatistics(2)  # 2 = wdStatisticPages
    doc.Close(False)
    print(f"PDF exported to tmp: pages={n_pages}, size={os.path.getsize(TMP)/1e6:.1f} MB")
finally:
    word.Quit()

try:
    shutil.move(TMP, OUT)
    print(f"replaced: {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
except PermissionError:
    print(f"[警告] 目标 PDF 被占用（可能正在阅读器中打开），临时文件保留在: {TMP}")
    print("请关闭 PDF 后手动替换，或告诉我再试一次。")
