# -*- coding: utf-8 -*-
"""Regenerate all manuscript figures (fig1-fig14) from current data.

Every `fig.savefig("*.png")` additionally emits a same-name vector PDF, so the
whole set gets PNG (600 dpi) + PDF with one pass. Run with the Anaconda python
(has matplotlib/pandas/scipy/tifffile).
"""
from __future__ import annotations

import os
import runpy
import sys
import time

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

SCRIPTS = [
    "redraw_fig1.py",            # fig1
    "redraw_figs_pub.py",        # fig2-fig8
    "plot_ladder_mechanism.py",  # fig9 + fig10
    "plot_decision_impact.py",   # fig11
    "plot_transfer.py",          # fig12
    "plot_shift_spectrum.py",    # fig13
    "plot_cliff_pair.py",        # fig14
]

_orig_savefig = Figure.savefig
_made: list[str] = []


def _savefig_both(self, fname, *a, **kw):
    _orig_savefig(self, fname, *a, **kw)
    if isinstance(fname, str) and fname.lower().endswith(".png"):
        pdf = fname[:-4] + ".pdf"
        _orig_savefig(self, pdf, *a, **{**kw, "format": "pdf"})
        _made.append(pdf)


def main() -> None:
    Figure.savefig = _savefig_both
    failures = []
    for s in SCRIPTS:
        p = os.path.join("scripts", s)
        t0 = time.time()
        try:
            runpy.run_path(p, run_name="__main__")
            print(f"[ok] {s} ({time.time()-t0:.1f}s)", flush=True)
        except SystemExit:
            print(f"[ok] {s} (SystemExit)", flush=True)
        except Exception as e:  # noqa: BLE001
            failures.append(s)
            print(f"[FAIL] {s}: {type(e).__name__}: {e}", flush=True)
    Figure.savefig = _orig_savefig
    print("\n== PDFs written in this pass ==")
    for m in sorted(set(_made)):
        print("  ", m)
    if failures:
        print("\nFAILED:", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
