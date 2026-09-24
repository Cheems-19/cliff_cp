# -*- coding: utf-8 -*-
"""Graphical abstract 草版：左 = 悬崖对（失效），右 = 转移诊断（可部署判据）。
定刊后按刊尺寸要求重排（改 TARGET_W）。"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 600,
                     "savefig.bbox": "tight"})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

A = mpimg.imread("figures/fig14_cliff_pair.png")
B = mpimg.imread("figures/fig12_transfer_diagnostic.png")

fig = plt.figure(figsize=(11.0, 3.8))
axA = fig.add_axes([0.005, 0.06, 0.47, 0.80])
axB = fig.add_axes([0.520, 0.06, 0.475, 0.80])
axA.imshow(A)
axB.imshow(B)
axA.axis("off")
axB.axis("off")

axA.set_title("Conformal failure concentrates on activity cliffs",
              fontsize=12, fontweight="bold", pad=6)
axB.set_title("A label-free diagnostic predicts it before deployment",
              fontsize=12, fontweight="bold", pad=6)

out = "figures/graphical_abstract_draft.png"
fig.savefig(out)
print(f"[saved] {out}")
