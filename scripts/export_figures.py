# -*- coding: utf-8 -*-
"""导出投稿用图件：figures_submission/

做法（不改动 5 个绘图脚本）：
  1) monkeypatch `Figure.savefig`，使脚本里的 `fig.savefig("*.png")` 额外写出同名 .pdf
     —— 从而一次性得到 12 张**矢量 PDF**（线稿首选）与 12 张 600 dpi PNG；
  2) 把 PNG 转成 **TIFF (LZW)**，宽度归一到双栏 17.4 cm（仅当原图更宽时下采样，
     不做上采样以免失真），并写入有效的 dpi；
  3) 生成 MANIFEST.csv / MANIFEST.md，列出每张图的目标栏宽、像素、有效 dpi 与建议位置。
"""
from __future__ import annotations

import os
import runpy
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
OUT = "figures_submission"
os.makedirs(OUT, exist_ok=True)

SCRIPTS = ["redraw_figs_pub.py", "redraw_fig1.py", "plot_decision_impact.py",
           "plot_transfer.py", "plot_ladder_mechanism.py"]

# ---- 1) 让绘图脚本顺带输出矢量 PDF ----
_orig_savefig = Figure.savefig
_made = []


def _savefig_both(self, fname, *a, **kw):
    _orig_savefig(self, fname, *a, **kw)
    if isinstance(fname, str) and fname.lower().endswith(".png"):
        pdf = fname[:-4] + ".pdf"
        _orig_savefig(self, pdf, *a, **{**kw, "format": "pdf"})
        _made.append(pdf)


if "--export-only" not in sys.argv:
    Figure.savefig = _savefig_both
    for s in SCRIPTS:
        p = os.path.join("scripts", s)
        try:
            runpy.run_path(p, run_name="__main__")
            print(f"[ok] {s}", flush=True)
        except SystemExit:
            pass
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {s}: {type(e).__name__}: {e}", flush=True)
    Figure.savefig = _orig_savefig

# ---- 2) PNG -> TIFF（zlib/Adobe-Deflate 无损；Pillow 的 TIFF 压缩在本机 12.3.0 会崩，故用 tifffile） ----
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from PIL import Image  # noqa: E402

DOUBLE_CM = 17.4
SINGLE_CM = 8.4
target_in = DOUBLE_CM / 2.54

rows = []
figs = sorted([f for f in os.listdir("figures") if f.startswith("fig") and f.endswith(".png")],
              key=lambda x: int("".join(ch for ch in x.split("_")[0] if ch.isdigit())))
for f in figs:
    num = int("".join(ch for ch in f.split("_")[0] if ch.isdigit()))
    src = os.path.join("figures", f)
    im = Image.open(src).convert("RGB")
    w, h = im.size
    natural_in = w / 600.0
    if natural_in > target_in:                      # 只下采样
        new_w = int(round(target_in * 600))
        new_h = int(round(h * new_w / w))
        im = im.resize((new_w, new_h), Image.LANCZOS)
        eff_dpi = new_w / target_in
        placement = "double column"
    else:
        eff_dpi = 600.0
        placement = "double column (natural size)" if natural_in > SINGLE_CM / 2.54 else "single column"
    stem = f"Fig{num}_" + f.split("_", 1)[1].rsplit(".", 1)[0]
    tif = os.path.join(OUT, stem + ".tif")
    tifffile.imwrite(tif, np.asarray(im), photometric="rgb", compression="zlib",
                     resolution=(round(eff_dpi), round(eff_dpi)), resolutionunit="INCH")
    # 矢量 PDF（若已生成）
    pdf_src = os.path.join("figures", f[:-4] + ".pdf")
    if os.path.exists(pdf_src):
        shutil.copy2(pdf_src, os.path.join(OUT, stem + ".pdf"))
    rows.append(dict(figure=f"Figure {num}", tif=os.path.basename(tif),
                     pdf=(stem + ".pdf") if os.path.exists(pdf_src) else "",
                     px=f"{im.size[0]}x{im.size[1]}",
                     width_cm=round(im.size[0] / eff_dpi * 2.54, 2),
                     eff_dpi=int(round(eff_dpi)), placement=placement,
                     tif_kb=round(os.path.getsize(tif) / 1024)))

import csv  # noqa: E402
with open(os.path.join(OUT, "MANIFEST.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wtr.writeheader()
    wtr.writerows(rows)

with open(os.path.join(OUT, "MANIFEST.md"), "w", encoding="utf-8") as fh:
    fh.write("# 投稿图件清单（figures_submission/）\n\n")
    fh.write("规格：TIFF（**zlib / Adobe Deflate 无损压缩**，经 tifffile 写出）目标双栏宽 **17.4 cm**；"
             "**仅下采样，不上采样**。\n")
    fh.write("另附**矢量 PDF**（由绘图脚本直接输出，线稿与文字可无损缩放，多数期刊首选）。\n\n")
    fh.write("| 图 | TIFF | 大小 | 矢量 PDF | 像素 | 宽 (cm) | 有效 dpi | 建议栏位 |\n"
             "|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        fh.write(f"| {r['figure']} | `{r['tif']}` | {r['tif_kb']} KB | "
                 f"{'`'+r['pdf']+'`' if r['pdf'] else '—'} | "
                 f"{r['px']} | {r['width_cm']} | {r['eff_dpi']} | {r['placement']} |\n")
    fh.write("\n> 说明：所有图均为 600 dpi 重绘；TIFF 已按双栏宽归一化（有效 dpi 见上表）。\n")
    fh.write("> 若目标刊要求单栏（8.4 cm）或指定宽度，改 `DOUBLE_CM` 后以 "
             "`--export-only` 重跑本脚本即可（跳过重绘）。\n")
    fh.write("> 注：本机 Pillow 12.3.0 的压缩 TIFF 写出会崩溃，故 TIFF 由 tifffile 生成；"
             "LZW 需 `imagecodecs`，未安装时用 zlib（期刊同样接受）。\n")

print(f"\n[export] {len(rows)} figures -> {OUT}/")
for r in rows:
    print(f"  {r['figure']:10s} {r['px']:>11s}  {r['width_cm']:>5}cm  {r['eff_dpi']:>3d}dpi  {r['placement']}")
