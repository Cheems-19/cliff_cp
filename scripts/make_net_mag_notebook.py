# -*- coding: utf-8 -*-
"""生成 notebooks/net_mag_demo.ipynb（代码单元与 test_net_mag_demo.py 同源）。"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "notebooks", "net_mag_demo.ipynb")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s.splitlines(keepends=True)}


cells = [
    md("""# `net_mag` in 5 minutes — a label-free pre-deployment diagnostic

Given a **calibration set** (scores + signal values) and a candidate **Mondrian partition
signal**, `net_mag` predicts whether the signal will *harm* a subgroup you care about
(activity cliffs, a priority scaffold, a toxic substructure) **without touching any test
label** — see the manuscript, Proposition 1 and SI §S5.2.

```
net_mag(G; s) = E_{i∈G}[ max(0, (q_{k(i)} − q̄)/q̄) ] − E_{i∈G}[ max(0, (q̄ − q_{k(i)})/q̄) ]
```

* `net_mag < 0` → the subgroup is **net-tightened** → expect a coverage **loss** under this signal;
* `net_mag > 0` → net-widened → expected benefit;
* `|net_mag|` near zero → no practically meaningful effect
  (on all 16 signal×representation units of the study, sign(`net_mag`) = sign(ΔCov);
  |net_mag| < 0.5 in percent-of-q̄ units indicated no practical effect).

This notebook is **NumPy-only** and runs in ~1 minute on CPU; it needs no dataset download."""),
    code("""import sys, os
# if you cloned the repo: make src/ importable (skip if installed via pip)
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

import numpy as np
from cliffcp.diagnostics import net_mag, conformal_quantile, mondrian_quantiles

rng = np.random.default_rng(20260924)
ALPHA = 0.10"""),
    md("""## 1. A synthetic calibration set with a *misaligned* signal

We simulate nonconformity scores |y − ŷ| whose difficulty **co-varies with the signal**
(low signal ⇒ genuinely easy molecules on average). This is the healthy case: on the
calibration set the signal honestly explains difficulty, so the lowest-signal bin gets a
**small** conformal quantile (its bin is *tightened*)."""),
    code("""n_cal = 4000
cal_sig    = rng.normal(size=n_cal) + 0.2 * rng.normal(size=n_cal)   # the signal
cal_scores = rng.lognormal(mean=-2.0, sigma=1.0, size=n_cal) * np.exp(0.8 * cal_sig)

q_bar = conformal_quantile(cal_scores, ALPHA)
edges, q_bins, _, _ = mondrian_quantiles(cal_scores, cal_sig, n_bins=3, alpha=ALPHA)

print(f"global quantile q_bar = {q_bar:.4f}")
print(f"bin edges (signal)    = {np.round(edges, 3)}")
print(f"per-bin quantiles     = {np.round(q_bins, 4)}   <- lowest bin is tightened")"""),
    md("""## 2. The trap: a subgroup where the signal stops working

On new data, the 10% of molecules the signal deems *easiest* (lowest signal) are
**locally broken**: their scores are anomalously high (here: 1.6× inflated, mimicking an
activity-cliff neighbourhood where labels contradict the trend). The signal cannot see
this — it still assigns them to the tightened bin. This is exactly the misalignment
mechanism."""),
    code("""n_test = 4000
sig_test = rng.normal(size=n_test) + 0.2 * rng.normal(size=n_test)
scores_test = rng.lognormal(mean=-2.0, sigma=1.0, size=n_test) * np.exp(0.8 * sig_test)

cliff = sig_test <= np.quantile(sig_test, 0.10)          # the subgroup of interest
scores_test[cliff] *= 1.6                                 # local failure the signal misses

# ---- the whole diagnostic is ONE call ----
nm = net_mag(
    target_signal=sig_test,     # signal values for the molecules you care about
    cal_scores=cal_scores,      # |y - yhat| on the calibration set
    cal_signal=cal_sig,         # same signal on the calibration set
    group=cliff,                # boolean mask / indices of the subgroup
    n_bins=3, alpha=ALPHA,
)
print(f"net_mag(cliff) = {nm:+.3f}   -> negative = harm warning")"""),
    md("""## 3. Validate against the realised coverage change (labels needed only HERE, for the demo)"""),
    code("""b = np.searchsorted(edges, sig_test, side="right")
cov_mond = scores_test <= q_bins[b]
cov_glob = scores_test <= q_bar
d_cliff = cov_mond[cliff].mean() - cov_glob[cliff].mean()
print(f"realised ΔCov(cliff) = {d_cliff*100:+.2f} pp   (Mondrian − global)")
print(f"sign(net_mag) == sign(ΔCov): {np.sign(nm) == np.sign(d_cliff)}")"""),
    md("""A strong misalignment (net_mag ≈ −0.7) is realised as a cliff coverage loss of several
percentage points, while the *marginal* coverage stays at ≈ 1 − α — the subgroup failure
that marginal guarantees cannot see."""),
    md("""## 4. Using it on **your** data

Replace the arrays below; nothing else changes. The signal can be *any* label-free
partition variable: local training-label dispersion (`d_wstd`), tree-ensemble variance
(`rf_std`), training-neighbour count (`d_nb`), …

```python
nm, details = net_mag(
    target_signal=my_signal_test,   # (n_target,) float
    cal_scores=my_scores_cal,       # (n_cal,)   |y - yhat| on calibration
    cal_signal=my_signal_cal,       # (n_cal,)   same signal, calibration molecules
    group=cliff_mask,               # bool mask or int indices over the n_target molecules
    n_bins=3,                       # manuscript recommends 3 or 5
    alpha=0.10,
    return_details=True,            # -> q_bar, bin edges, per-bin quantiles, per-molecule contributions
)
# nm < 0  -> do NOT conditionalise on this signal (coverage loss expected for `group`)
# nm > 0  -> expected benefit;  |nm|*100 < 0.5 -> no practical effect
```"""),
    md("""## Citation

If this diagnostic is useful to you, please cite the manuscript and the software record
(see `CITATION.cff`). MIT licence."""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("written:", OUT)

# —— 验证：把所有 code 单元按顺序在一个进程里跑一遍 ——
# 模拟 Jupyter：cwd = notebook 所在目录（sys.path 的 ".." 相对它解析）
import io, contextlib
os.chdir(os.path.dirname(OUT))
ns = {"__file__": OUT}
ok = True
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, f"cell{i}", "exec"), ns)
        print(f"cell{i}: OK\n" + buf.getvalue())
    except Exception as e:
        ok = False
        print(f"cell{i}: FAIL -> {type(e).__name__}: {e}")
print("\n[notebook cells all executed]" if ok else "\n[NOTEBOOK EXECUTION FAILED]")
