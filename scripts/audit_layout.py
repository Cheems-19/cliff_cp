# -*- coding: utf-8 -*-
"""图片排版审计：逐个运行绘图脚本，在 savefig 前拦截，
计算所有 Text 艺术家的窗口包围盒，报告：
  A) 文本两两重叠（重叠面积 > 25% 较小者）
  B) 文本超出画布边界（可能被裁切）
输出：每张图的审计结论。
"""
from __future__ import annotations

import os
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.text as mtext  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

SCRIPTS = ["redraw_figs_pub.py", "redraw_fig1.py", "plot_decision_impact.py",
           "plot_transfer.py", "plot_ladder_mechanism.py", "plot_shift_spectrum.py",
           "plot_cliff_pair.py"]

_orig_savefig = Figure.savefig


def _collect_texts(fig):
    items = []
    for ax in fig.axes:
        arts = list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        if ax.axison:  # axis("off") 的面板：刻度标签不参与（默认刻度是假阳性）
            # 只保留视图区间内的刻度（视图外的刻度标签不渲染，是假阳性）
            ylim = ax.get_ylim()
            ytl = [tl for tl, tv in zip(ax.get_yticklabels(), ax.get_yticks())
                   if min(ylim) - 1e-9 <= tv <= max(ylim) + 1e-9]
            xlim = ax.get_xlim()
            xtl = [tl for tl, tv in zip(ax.get_xticklabels(), ax.get_xticks())
                   if min(xlim) - 1e-9 <= tv <= max(xlim) + 1e-9]
            arts += xtl + ytl
        leg = ax.get_legend()
        if leg is not None:
            arts += leg.get_texts()
        for t in arts:
            if not (isinstance(t, mtext.Text) and t.get_visible() and t.get_text().strip()):
                continue
            items.append(t)
    return items


def _audit(fig, tag):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = _collect_texts(fig)
    boxes = []
    seen_pos = set()
    for t in items:
        try:
            bb = t.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if not np.isfinite(bb.x0) or not np.isfinite(bb.y0):
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        key = (t.get_text(), round(bb.x0), round(bb.y0))  # twin 轴重复刻度去重
        if key in seen_pos:
            continue
        seen_pos.add(key)
        boxes.append((t.get_text().replace("\n", "⏎")[:28], bb))

    problems = []
    # A) 两两重叠
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            t1, b1 = boxes[i]
            t2, b2 = boxes[j]
            ix = max(0, min(b1.x1, b2.x1) - max(b1.x0, b2.x0))
            iy = max(0, min(b1.y1, b2.y1) - max(b1.y0, b2.y0))
            inter = ix * iy
            smaller = min(b1.width * b1.height, b2.width * b2.height)
            if smaller > 0 and inter / smaller > 0.25:
                problems.append(f"OVERLAP  [{t1}] × [{t2}]  ({inter / smaller:.0%})")
    # B) 超出画布
    W, H = fig.canvas.get_width_height()
    for txt, bb in boxes:
        if bb.x0 < -2 or bb.y0 < -2 or bb.x1 > W + 2 or bb.y1 > H + 2:
            problems.append(f"OUTSIDE [{txt}]  bbox=({bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f}) canvas=({W}x{H})")
    return problems


def patched_savefig(self, fname, *a, **kw):
    if isinstance(fname, str) and fname.lower().endswith(".png"):
        probs = _audit(self, fname)
        if probs:
            print(f"\n### AUDIT {os.path.basename(fname)} —— {len(probs)} 处问题")
            for p in probs[:12]:
                print("   ", p)
        else:
            print(f"OK  {os.path.basename(fname)}")
    return _orig_savefig(self, fname, *a, **kw)


Figure.savefig = patched_savefig

for s in SCRIPTS:
    p = os.path.join("scripts", s)
    print(f"\n══════ {s} ══════")
    try:
        runpy.run_path(p, run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:
        print(f"  [脚本异常] {type(e).__name__}: {e}")
