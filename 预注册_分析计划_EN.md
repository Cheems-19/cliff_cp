# Pre-registered analysis plan

> **Purpose.** To *fix* hypotheses, statistical conventions and decision rules before submission, so that
> significant results cannot be selected post hoc (p-hacking / HARKing). Every claim in the manuscript is
> stratified and adjudicated according to this document; any analysis outside it is labelled exploratory.
> **Version:** v1 · 2026-09-23. Once frozen, any modification must be recorded in the change log together
> with its justification and date.
>
> *This is the English translation. The original Chinese-language registration document
> (`预注册_分析计划.md`) is distributed with the code repository and is the version of record for the
> freeze date.*

---

## 0. Fixed settings (not searched over)

| Item | Value |
|---|---|
| Datasets | the 15 ChEMBL single-target datasets in MoleculeACE satisfying N ≥ 800 and ≥ 60 dataset-level cliff molecules |
| Representation | predictor features ECFP4-2048 or RDKit 2D descriptors (~210 dims); **cliff determination and neighbourhood retrieval always use ECFP4** |
| Split | random (primary); scaffold (Bemis–Murcko, additional analysis) |
| Split ratio | train/cal/test = 60/20/20 |
| Nominal coverage | α = 0.10 (primary); α ∈ {0.07, 0.10, 0.13, 0.16} (for matched coverage) |
| Cliff definitions | `cliffA` (dataset-level: ∃ sim ≥ 0.85 and \|ΔpKi\| ≥ 1.0); `cliffB` (mechanism: the higher-activity side of a pair established through *training* neighbours with sim ≥ 0.85 and \|Δy\| ≥ 1.0) |
| Model families | RF (200 trees), GBR (5-member multi-seed ensemble), D-MPNN (chemprop 2.3.1, 3 members) |
| Seeds | 10 for the main analysis; 4–5 for cross-architecture and sensitivity analyses (stated individually) |
| Direction consistency | "number of datasets with the correct direction / total"; **any conclusion below 15/15 must be quoted with its range** |

**Statistical conventions (fixed).** Cross-dataset pooling first pairs *within* (dataset, seed), then averages
over seeds within a dataset (with the standard error), and finally applies random-effects pooling
(reporting Δ, p and I²). Coverage intervals use Wilson. Coverage backtests use a **split-level z-test**
(not a within-split binomial test).

---

## 1. Primary (confirmatory) hypotheses

> Decision rule: random-effects pooled p < 0.05 **and** direction consistency ≥ 13/15. Multiple comparisons
> are controlled within the primary family (4 tests) by Holm–Bonferroni.

| ID | Hypothesis | Statistic | Decision |
|---|---|---|---|
| **P1** | Cliff-group marginal coverage is below that of non-cliff compounds | Δ coverage (non-cliff − cliff), paired Wilcoxon | p < 0.05 and ≥ 13/15 |
| **P2** | In-domain cliff compounds fail more often than in-domain non-cliff compounds | failure-rate ratio (2 AD families × 4 thresholds = 8 definitions) | 8/8 direction-consistent and ratio > 1.5 |
| **P3** | Conditioning on a **misaligned signal** (`d_nb`, `rf_std`) lowers cliff-group coverage below global | Δ coverage, paired Wilcoxon, matched-coverage normalisation | p < 0.05 and ≤ 4/15 in the reverse direction |
| **P4** | Conditioning on the **aligned signal** (`d_wstd`) does not harm the cliff group (direction consistently positive) | Δ coverage, paired Wilcoxon + direction consistency | direction consistency ≥ 10/15 (significance not required) |

> P4's weaker decision rule is **declared in advance**: the effect is small and power is limited (see §4),
> so "does not harm" is adjudicated by direction consistency rather than by significance.

---

## 2. Secondary hypotheses

| ID | Hypothesis | Criterion |
|---|---|---|
| S1 | Representation axis reproduces (ECFP4 → RDKit 2D, 8 partition variables with consistent signs) | 8/8 signs consistent |
| S2 | Model-family axis reproduces (RF → GBR; `d_nb` harmful, `d_wstd` safe) | both families agree in direction |
| S3 | `rf_std` is harmless on D-MPNN (its ensemble variance has small misalignment) | direction agrees with the misalignment ordering |
| S4 | Coverage backtests: every method's marginal coverage is within ±0.3 pp of nominal | split-level z-test |
| S5 | Method-ladder ordering agrees with the misalignment prediction (CQR < global < misaligned signals) | ordering consistent |
| S6 | **P1 still holds under scaffold splitting** (split realism) | random-effects pooled p < 0.05 |
| S7 | **Adaptive / weighted CP do not remove the cliff gap** (baseline completeness) | their Δ cliff coverage vs global is not significantly positive |
| S8 | **Decision impact**: the share of true active cliff compounds that are under-estimated exceeds that of non-cliff compounds | per-dataset pairing, direction consistency ≥ 13/15 |

> S6–S8 were added in v1 (responding to the methodological attack surfaces a reviewer may raise). The
> **expected direction** for S7 was: adaptive/weighted improve marginal coverage or width but **do not
> specifically repair the cliff group**; if the measurement contradicted this, it was to be reported
> faithfully and the conclusions revised.

---

## 3. Exploratory (not used for primary conclusions)

- The quantitative relation between misalignment and harm (including `r = −0.63`) and the `d_nn_sim` counterexample
- The +1.2 pp of `d_wstd` under matched coverage (p ≈ 0.099)
- The effect of increasing GNN ensemble members from 3 to 8 (known confounding; observational only)
- Details of the `n_bins` / `sim` / `act` threshold sweeps
- **The clean mechanism test**: signal-precision ladder with a frozen predictor (RF tree-subset SD)
- The τ sweep and budget curves of the decision-impact analysis
- **The coverage-transfer identity (Proposition 1) and the label-free diagnostic `net_mag` (Eq. 5)**: the
  identity is exact (distribution-free), while its empirical validation (ρ = 0.865, n = 16) rests on runs
  already completed and is therefore a **post-hoc verification**; it is presented as "theory + validation"
  and makes no primary significance claim.

---

## 4. Obligation to report counterexamples and failures

1. `d_nn_sim`: largest misalignment yet no harm → misalignment is a **screening rule, not a quantitative formula**.
2. `d_iqr`: misalignment ≈ 0 yet ineffective → "misalignment ≈ 0" is **necessary but not sufficient**.
3. At `act = 1.5` the `rf_std` conclusion reverses (only 9 datasets) → its harmfulness is bounded to `act ∈ {0.5, 1.0}`.
4. The `sim = 0.95` configuration is unusable (only 1 of 15 datasets survives) → recorded as task sparsity.
5. The GNN uses only 3 ensemble members with I² = 79% → recorded as a limitation.

---

## 5. Multiple comparisons and reporting standards

- The **primary family** (P1–P4) uses Holm–Bonferroni; adjusted q-values are reported.
- **Secondary/exploratory** results report effect sizes + 95% CIs + direction consistency, and **claim no significance**.
- All coverage gaps are reported with Wilson intervals; all pooled quantities with I².
- Every number must be regenerable by a single script (script paths are given in the appendix).
- The "best" value of a threshold must not be reported as a conclusion (`n_bins` is given only as a recommended range, 3 or 5).

---

## 6. Verdicts for S6–S8 (recorded 2026-09-23)

| ID | Decision rule | Observed | Verdict |
|---|---|---|---|
| **S6** | P1 still holds under scaffold splitting: random-effects pooled p < 0.05 | scaffold · `cliffA`: Δ = +5.0 pp, direction consistency 13/15, Wilcoxon p = 4.3×10⁻⁴ (random · `cliffA`: +9.5 pp, 14/15; random · `cliffB`: +17.3 pp, 15/15; scaffold · `cliffB`: +8.9 pp, 12/15) | ✅ **holds**, but the **effect halves** (9.5 → 5.0 pp) and must be reported faithfully as "partly split-dependent" |
| **S7** | Adaptive / weighted do not remove the cliff gap (their Δ vs global is not significantly positive) | residual gap 6.8 pp after the best baseline, CQR (random +2.73 pp); **adaptive: random +2.54 pp, p = 0.055, at a +21% width cost**; weighted: −1.26 pp random / −1.94 pp scaffold | ✅ **holds** (the gap is not removed), but the **expected direction was partly contradicted** — see below |
| **S8** | Decision impact: under-estimation of true active cliff compounds exceeds that of non-cliff compounds, direction consistency ≥ 13/15 | 35.3% vs 13.3% (+22.0 pp, 2.66×), 15/15 | ✅ holds (reported in §3.7 of the main text) |

### ⚠️ A pre-registered expectation was partly contradicted (must be reported faithfully)
The v1 expectation for S7 was that adaptive / weighted CP would "improve marginal coverage or width but **not
specifically repair the cliff group**; if the measurement contradicted this, report it faithfully and revise
the conclusions". The measurement shows that **adaptive CP did specifically improve the cliff group**
(random split: +2.54 pp; mechanism-definition `cliffB`: +6.37 pp, 13/15, p = 3.4×10⁻³), which is the
**opposite** of that expectation. Handling:
1. §3.8 of the main text states this explicitly and notes that **adding this baseline was necessary** —
   otherwise "no method other than CQR helps" would have been an overstatement;
2. its **cost** (+21% / +28% width) and the **disappearance of the gain under scaffold splitting** (−0.34 pp)
   are reported alongside, so the main conclusions are unchanged;
3. this entry is recorded here **without modifying the original decision rules**, which remain as frozen
   before the data were seen.

### Implementation defect disclosed (affecting two rows of Table 4)
On the first run of phase11 the **Mondrian ← `d_wstd` and Mondrian ← `d_nb` arms silently degenerated to the
global method** (Δ exactly 0.00 pp, direction consistency 0/15, Wilcoxon p = nan).
Root cause: `dispersion.neighbor_label_stats` leaves `d_wstd` and others as NaN for molecules with **no
training neighbour** (the no-neighbour branch sets only `d_nb` and `d_pair_gap`), while `bins_by_cal` passed
the NaN-containing array straight to `np.quantile`, so every bin edge became NaN and `searchsorted` placed all
molecules in a single bin; that bin necessarily met the minimum-size rule and the arm fell back to the global
quantile.
Fix: apply the main pipeline's established convention (NaN → sparsest bin, i.e. NaN → 0), add the missing
`mondrian_nb` arm, and re-run. Table 4 reports the re-run numbers.

**Re-run results (completed 2026-09-23 17:12, rows = 600 = 15 × 2 × 5 × 4 ✓):**
Mondrian ← `d_wstd` (aligned) Δ = **+0.36 pp** (random, 9/15, p = 0.36) / **−0.71 pp** (scaffold, 5/15, p = 0.19);
Mondrian ← `d_nb` (misaligned) Δ = **−2.70 pp** (random, 3/15, p = 6.3×10⁻³) / **−3.82 pp** (scaffold, 2/15, p = 1.5×10⁻³).
→ After the fix, the **"aligned does not harm / misaligned harms significantly" contrast holds under both
splits**, in agreement with the misalignment prediction of §3.2 of the main text.
**Reproducibility cross-check:** the `mondrian_rfstd` coverage of the re-run and the first run is **identical
to the last decimal** on the same (split, dataset, seed) triplets (150 triplets, maximum absolute difference
0.000), confirming that the two runs are deterministically reproducible and the rows can be spliced into the
same table.
**Nature of the change:** an implementation fix; it **does not alter any decision rule or any other arm's numbers**.

---

## 7. Changes made outside this plan

| Date | Change | Reason |
|---|---|---|
| 2026-09-23 | Added S6–S8 (scaffold / adaptive-weighted / decision impact) | to cover three reviewer-facing attack surfaces: split realism, baseline completeness, decision impact |
| 2026-09-23 | Abstract coverage-gap figure changed from "20–24 pp" to **18.2 pp** (per-dataset 7.7–32.4, 15/15) | the original value had no data support; the measured macro value at α = 0.10 is 18.2 pp |
| 2026-09-23 | Added theoretical subsection §2.3.1 (coverage-transfer identity + `net_mag`), listed under §3 exploratory | post-hoc derivation and verification; makes no primary claim |
| 2026-09-23 | phase11/13 completed and §3.8 filled in (Tables 4 and 5); S6/S7/S8 verdicts in §6 | all three secondary hypotheses of the pre-registration now have a verdict |
| 2026-09-23 | Fixed the phase11 Mondrian ← `d_wstd` / `d_nb` implementation defect (NaN binning collapse → silent degeneration to global) and filled in the re-run numbers | the first run showed Δ ≡ 0.00 pp and p = nan on that arm, verified to be an **artefact rather than a zero effect**; an implementation fix that changes no decision rule |
| 2026-09-23 | Recorded that the S7 expected direction was partly contradicted (adaptive does improve the cliff group) | when a pre-registered expectation is contradicted, report it faithfully and keep the original criterion |
