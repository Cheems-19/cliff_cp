# cliff-cp

**Code, data splits, and per-molecule records for:** *How the coverage guarantee of conformal prediction fails on activity cliffs: mechanism, diagnosis, and a structural blind spot of applicability domains.*

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> ⬆️ Replace the Zenodo badge/DOI after the first release (see *Archiving* below).

---

## What this repository shows

Conformal prediction (CP) gives molecular property models a finite-sample **marginal** coverage guarantee — but "marginal" means *on average*. Using 15 ChEMBL single-target datasets from the [MoleculeACE](https://github.com/molML/MoleculeACE) benchmark (70,170 test-molecule records, 3 model families, 2 molecular representations, 7 threshold configurations), this work shows:

| Finding | Result |
|---|---|
| **CP failure is concentrated on activity cliffs** | cliff compounds are covered **18.2 pp** less often than non-cliff compounds (7.7–32.4 pp across datasets; 15/15 direction-consistent) |
| **The mechanism is signal misalignment** | a signal that "sees cliffs" but not "failure" transfers coverage *away* from the cliff group under Mondrian conditionalization |
| **A label-free, pre-deployment diagnostic** | an exact distribution-free **coverage-transfer identity** yields `net_mag`, which predicts which signals will harm a target subgroup (Spearman ρ = 0.865, n = 16) |
| **Applicability domains are structurally blind** | 40.8–75.9% of CP failures occur *inside* the AD; in-domain cliff compounds fail **3.4–5.3×** more often than in-domain non-cliff compounds (all 8 domain definitions) |
| **Downstream cost** | among the most potent true actives, **35%** of cliff compounds receive an interval whose upper bound falls below their true potency, versus 13% of non-cliff compounds |

**Scope note.** We do **not** propose a new interval constructor — CQR remains stronger at closing the gap (8.6 pp residual). The contribution is a mechanism, a pre-deployment criterion, and a quantified boundary.

---

## Repository layout

```
cliff-cp/
├── src/cliffcp/                 # core library
│   ├── data.py                  #   source-agnostic dataset loading (MoleculeACE-compatible)
│   ├── features.py              #   ECFP4 + exact Tanimoto neighbourhoods
│   ├── cliffs.py                #   three activity-cliff definitions (cliffA / cliffB / strong)
│   ├── dispersion.py            #   local label-dispersion signals
│   ├── conformal.py             #   global / Mondrian / cluster / kNN-weighted / adaptive / weighted-shift
│   ├── stats.py                 #   Wilson intervals, stratified permutation, group coverage
│   ├── diagnostics.py           #   net_mag: label-free pre-deployment diagnostic (NumPy-only)
│   └── gnn.py                   #   D-MPNN (chemprop) integration
├── scripts/                     # analysis, aggregation and plotting entry points
├── notebooks/
│   └── net_mag_demo.ipynb       # runnable 5-minute demo of the net_mag diagnostic
├── results_*/                   # aggregated tables (CSV) + dump/ per-molecule records
├── figures/                     # manuscript figures (PNG)
├── figures_submission/          # 600 dpi TIFF + vector PDF + MANIFEST (journal-ready)
├── requirements.txt             # frozen dependency set
├── 预注册_分析计划.md            # public OSF preregistration supplement (full plan, mirrored in SI §S7)
├── CITATION.cff                 # machine-readable citation metadata
└── README.md                    # this file
```

---

## Installation

Python 3.10+ recommended.

```bash
git clone https://github.com/{user}/cliff-cp.git
cd cliff-cp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used for the published results (numpy, pandas, scikit-learn, scipy, matplotlib, rdkit). The D-MPNN arm additionally needs `torch` and `chemprop` (GPU recommended); those are commented out by default so the CPU-only analyses install cleanly.

## Quickstart: the `net_mag` diagnostic (no data download needed)

```python
import sys; sys.path.insert(0, "src")
from cliffcp.diagnostics import net_mag

nm = net_mag(
    target_signal=signal_test,   # label-free signal for the molecules you care about
    cal_scores=cal_scores,       # |y - yhat| on your calibration set
    cal_signal=signal_cal,       # same signal on the calibration set
    group=cliff_mask,            # bool mask / indices of the subgroup of interest
    n_bins=3, alpha=0.10,        # manuscript recommends 3 or 5 bins
)
# nm < 0  -> this signal is expected to LOWER coverage for `group` (do not condition on it)
# nm > 0  -> expected benefit;  |nm|*100 < 0.5 -> no practical effect
```

NumPy-only — no RDKit required, drops into any pipeline. A runnable 5-minute walkthrough
lives in [`notebooks/net_mag_demo.ipynb`](notebooks/net_mag_demo.ipynb).

## Data

All datasets come from the public **MoleculeACE** benchmark:

```bash
git clone https://github.com/molML/MoleculeACE
```

Place the CSVs under `data/MoleculeACE/`. The exact splits used here are reproduced deterministically from the fixed seeds recorded in each script (see the pre-registered plan for the seed list), so no split files need to be downloaded separately.

---

## Reproducing the results

### Figures

| Figure | Contents | Command |
|---|---|---|
| Figure 1 | cliff vs non-cliff coverage; random-effects forest plot | `python scripts/redraw_fig1.py` |
| Figures 2–8 | mechanism, ablation, width–coverage frontier, AD blindness, AD sensitivity, representation axis, threshold heatmap | `python scripts/redraw_figs_pub.py` |
| Figures 9, 10 | method ladder; misalignment-vs-harm scatter | `python scripts/plot_ladder_mechanism.py` |
| Figure 11 | decision impact (under-estimation of true actives) | `python scripts/plot_decision_impact.py` |
| Figure 12 | coverage-transfer diagnostic (`net_mag` vs realised Δcoverage) | `python scripts/plot_transfer.py` |
| Submission package | 600 dpi TIFF + vector PDF + MANIFEST | `python scripts/export_figures.py` |

All figure scripts are deterministic and re-run from the CSV tables committed in `results_*/`; none require re-training a model.

### Tables

| Table | Command |
|---|---|
| Table 2 — matched-coverage comparison (paired Wilcoxon) | `python scripts/coverage_matched.py` |
| Table 3 — method ladder | aggregated from `results_phase1/` via `python scripts/meta_analysis.py` |
| Table 4 — random vs scaffold split × method | `python scripts/phase11_shift.py` → `python scripts/phase11_summarize.py` |
| Table 5 — signal-precision ladder (frozen predictor) | `python scripts/phase13_signal_ladder.py` |

### Full pipeline (from raw data)

```bash
# 1. per-dataset conformal runs over 15 datasets x seeds x methods
python scripts/phase1_signal.py --data-dir data/MoleculeACE --out results_phase1 --seeds 10

# 2. misalignment and within-group discrimination
python scripts/phase2_misalignment.py        # -> results_phase2/
python scripts/analyze_misalignment_relation.py

# 3. shift-intensity spectrum: random vs Bemis-Murcko scaffold split,
#    with adaptive/normalized and covariate-shift weighted CP baselines
python scripts/phase11_shift.py --data-dir data/MoleculeACE --out results/phase11_shift \
    --seeds 5 --split both

# 4. controlled mechanism test: freeze the predictor, vary only the signal's precision
python scripts/phase13_signal_ladder.py

# 5. decision impact (pure post-processing of the per-molecule dumps; no re-training)
python scripts/phase12_decision_impact.py

# 6. coverage-transfer identity + label-free diagnostic
python scripts/phase14_transfer_identity.py   # -> results_phase14/
```

Every step writes its own `SUMMARY.txt` alongside the CSV outputs.

---

## Statistical conventions

- Coverage differences are pooled with a **random-effects meta-analysis**; pooled Δ, pooled p and I² are always reported.
- Per-dataset **direction consistency** ("datasets with Δ>0 / total") is reported alongside every pooled number; any conclusion below 15/15 is quoted with its range.
- All coverage gaps carry **Wilson intervals**.
- Coverage backtests use a **split-level z-test** as the formal test; the binomial null is *not* valid within a single split (test molecules share a model and are positively correlated).
- Primary hypotheses are **Holm–Bonferroni** corrected within the primary family. The split between confirmatory, secondary and exploratory hypotheses is fixed in the pre-registered plan, and post-hoc additions are logged there.

---

## Licence and citation

Released under the **MIT licence** (see `LICENSE`).

If you use this code or data, please cite both the archived software record and the paper. `CITATION.cff` provides machine-readable metadata — GitHub shows a "Cite this repository" button generated from it.

```bibtex
@software{cliffcp2026,
  title  = {cliff-cp: coverage failure of conformal prediction on activity cliffs},
  author = {Tao, Shaobo},
  year   = {2026},
  doi    = {10.5281/zenodo.XXXXXXX}
}
```

## Archiving

1. Author/affiliation fields of `CITATION.cff` and `.zenodo.json` are filled (single author: Tao, Shaobo). Update both files if co-authors are added later.
2. Create a public OSF project and upload `预注册_分析计划.md` as the preregistration supplement; link it from the manuscript's Data/Code availability section.
3. Link the GitHub repository to Zenodo, then create a GitHub Release (tag e.g. `v1.0.0`).
4. Zenodo mints a DOI automatically — paste it back into this README, `CITATION.cff`, and the manuscript's Data/Code availability statements.

## Contact

Questions, corrections and replication attempts are welcome — please open an issue, or email **487455429@qq.com**.
