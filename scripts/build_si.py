# -*- coding: utf-8 -*-
"""组装 Supporting Information：SI_v1.md
数据源全部来自 results_*/ 的现成 CSV/CSV 与 预注册_分析计划.md。

!! 警告（2026-09-23）：SI_v1.md 自本生成器运行后已大量手工维护（S9/S10 重写、
!! B6 v2 数据 10 种子 x 5 基准、图注修订等），本脚本的模板已严重过时。
!! 重跑会用旧内容覆盖 SI_v1.md（第 621 行直接 write）。如需重新生成，
!! 必须先与现行 SI_v1.md 逐节核对，或改为输出到 SI_v1_GENERATED_DRAFT.md。
"""
import io, os, re
import pandas as pd
import numpy as np

C = r"C:/Users/Administrator/Desktop/小论文/cliff_cp"
OUT = []


def w(s=""):
    OUT.append(s)


TRANS = {
    "距离型 AD（1-NN 相似度）": "distance-type AD (1-NN similarity)",
    "密度型 AD（训练邻居数）": "density-type AD (no. of training neighbours)",
    "AD 规则（预警最疏的）": "AD rule (flag the sparsest)",
    "离散度规则（预警最分散的）": "dispersion rule (flag the highest dispersion)",
    "合并（两者都算预警）": "combined (either rule flags)",
    "组": "group",
    "AD 内 × 悬崖(cliffB)": "in-AD × cliff (cliffB)",
    "AD 内 × 非悬崖": "in-AD × non-cliff",
    "AD 外 × 非悬崖": "out-of-AD × non-cliff",
    "NaN→0（本文默认）": "NaN \u2192 sparsest bin (default)",
    "剔除": "drop molecule",
    "n_used": "n used",
    "n_dropped": "n dropped",
}


def md_table(df, floatfmt="%.4g", index=False):
    """DataFrame -> markdown 表格（转义竖线、英译分类标签）。"""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "OfU":
            df[c] = df[c].map(lambda v: TRANS.get(v, v))
    df.columns = [TRANS.get(str(c), str(c)) for c in df.columns]
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt % v)
    df = df.astype(str).replace({"nan": "—", "None": "—"})
    hdr = ("| " + " | ".join(str(c) for c in df.columns) + " |") if not index else \
          ("| " + str(df.index.name or "") + " | " + " | ".join(str(c) for c in df.columns) + " |")
    sep = "|" + "|".join(["---"] * (len(df.columns) + (1 if index else 0))) + "|"
    rows = []
    for idx, r in df.iterrows():
        cells = [str(v).replace("|", "\\|") for v in r.tolist()]
        if index:
            cells = [str(idx)] + cells
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([hdr, sep] + rows)


# ═══════════════════════ 封面 ═══════════════════════
w("# Supporting Information")
w()
w("**How the coverage guarantee of conformal prediction fails on activity cliffs: mechanism, diagnosis, and a structural blind spot of applicability domains**")
w()
w("*This file contains S1–S8: signal definitions and implementation, the full threshold-sensitivity table, "
  "the applicability-domain tables, the coverage backtests, the proof of Proposition 1 together with its "
  "empirical validation, the split-realism and signal-ladder results, the pre-registered analysis plan "
  "verbatim, and the matched-coverage method ladder.*")
w()
w("---")
w()

# ═══════════════════════ S1 ═══════════════════════
w("## S1  Conditioning signals: exact definitions and implementation")
w()
w("### S1.1  Neighbourhood construction")
w()
w("Molecules are represented as **ECFP4** (Morgan, radius 2, 2048 bits; RDKit). For a query molecule `i` we "
  "compute exact Tanimoto similarities to **all training molecules** (no approximate nearest-neighbour index), "
  "and define its neighbourhood as the `k` most similar training molecules whose similarity is at least "
  "`sim_floor`. Defaults: `sim_floor = 0.5`, `k = 50`. The global neighbour table used for the cliff "
  "definitions is built at `pair_sim = 0.85` with at most 512 neighbours per molecule, and is reused "
  "(a) to test whether two *training* neighbours are mutually similar and (b) to compute `d_pair_gap` without "
  "recomputing pairwise Tanimoto in Python.")
w()
w("### S1.2  Signal definitions")
w()
w("All eight signals depend only on the training set and on the query molecule's structure — **never on the "
  "query molecule's true label**. Writing the neighbourhood similarities as `s_1 ≥ s_2 ≥ … ≥ s_k` (the `k` "
  "selected training neighbours) and their labels as `y_1, …, y_k`:")
w()
w("| Signal | Definition | Side |")
w("|---|---|---|")
w("| `d_nb` | `\\|{ j : sim(i,j) ≥ sim_floor }\\|`, capped at `k` (local density) | data |")
w("| `d_wstd` | similarity-weighted SD: `sqrt( Σ s_j (y_j − μ)² / Σ s_j )`, with `μ = Σ s_j y_j / Σ s_j` | data |")
w("| `d_std` | unweighted SD of `{y_j}` | data |")
w("| `d_range` | `max{y_j} − min{y_j}` | data |")
w("| `d_iqr` | interquartile range of `{y_j}` (defined when `k ≥ 4`) | data |")
w("| `d_pair_gap` | `max` over neighbours `a` in the local neighbourhood of `max_b \\|y_b − y_a\\|` where `b` ranges over training molecules that are themselves neighbours of `a` at `pair_sim`; i.e. *does this region of chemical space contain a mutually-similar pair with a large label difference?* | data |")
w("| `d_nn_sim` | `s_1`, the similarity to the nearest training neighbour | data (distance) |")
w("| `rf_std` | model-side uncertainty: RF inter-tree SD / GBR multi-seed ensemble SD / D-MPNN ensemble SD | model |")
w()
w("`d_wstd` is the primary aligned signal: it is the only signal whose discriminative power for *cliffs* and "
  "for *failure* nearly coincide (§2.3 of the main text).")
w()
w("### S1.3  Missing-neighbour convention (and a bug it caused)")
w()
w("For molecules with **no** training neighbour above `sim_floor` (3,287 of 70,170 molecules, 4.68%), the "
  "statistics `d_wstd`, `d_std`, `d_range`, `d_iqr` and `d_nn_sim` are undefined and are stored as `NaN`; "
  "only `d_nb` and `d_pair_gap` are defined (both 0). The main analysis assigns these molecules to the "
  "**sparsest bin** (`NaN → 0`) and a `drop` variant is reported in S3.4; all conclusions hold in both.")
w()
w("> **Implementation note (disclosed).** In the first run of the split-realism experiment (S6), the "
  "`Mondrian ← d_wstd` arm silently degenerated to the global method because `NaN` values were passed "
  "directly to `numpy.quantile`, which made *every* bin edge `NaN`; `searchsorted` then placed all molecules "
  "in a single bin, that bin necessarily satisfied the minimum-size rule, and the arm fell back to the global "
  "quantile. The symptom was a Δ of exactly 0.00 pp with direction consistency 0/15 and a `NaN` Wilcoxon "
  "p-value. The cause was fixed by applying the convention above (NaN → sparsest bin) and the arm was re-run; "
  "S6 reports the re-run numbers. This affects **only** that arm — `d_nb` is explicitly zero-filled and "
  "`rf_std` contains no `NaN` — and no decision rule or other arm was changed.")
w()
w("### S1.4  Conformal methods compared")
w()
w("| Method | Construction |")
w("|---|---|")
w("| **global** (reference) | single split-conformal quantile `q̂` from all calibration scores |")
w("| **Mondrian ← signal** | calibration and test sets partitioned into `n_bins` groups by the **calibration-side** quantiles of the signal (`{20,40,60,80}%` cut points for `n_bins = 5`); a separate quantile per group; groups with fewer than 30 calibration points fall back to the global quantile |")
w("| **cluster** | k-means clustering of the calibration molecules; one quantile per cluster (localised calibration) |")
w("| **kNN-weighted** | each calibration score weighted by the Tanimoto similarity of its molecule to the test molecule (`k = 100` neighbours) |")
w("| **adaptive / normalised** | nonconformity scores normalised by an `x`-dependent scale `σ(x)` fitted on out-of-bag absolute residuals; intervals `ŷ ± q·σ(x)` |")
w("| **weighted (covariate shift)** | density ratio `w(x) = p_test(x)/p_cal(x)` estimated by a *calibration-versus-test* discriminator, with a weighted quantile of the calibration scores (Tibshirani et al. 2019) |")
w("| **CQR** (external competitor) | conformalised quantile regression, `HistGradientBoostingRegressor` with quantile loss, `α` and `1−α` heads, `max_iter = 400` |")
w()
w("### S1.5  Model families, representations and splits")
w()
w("| Item | Setting |")
w("|---|---|")
w("| Random forest | `n_estimators = 200`; inter-tree SD as model-side uncertainty |")
w("| Gradient boosting | `HistGradientBoosting`, 5-member multi-seed ensemble, `max_features = 0.4` (without injected randomness the ensemble variance collapses to zero and becomes a spurious signal) |")
w("| D-MPNN | chemprop 2.3.1, `d_h = 300`, `depth = 3`, 30 epochs, patience 5, 3-member ensemble |")
w("| Predictor representation | ECFP4 (2048 bits) **or** RDKit 2D descriptors (~210 dims), controlled separately from the cliff-detection representation |")
w("| Split (main) | random train/cal/test = 60/20/20, 10 seeds |")
w("| Split (robustness) | Bemis–Murcko scaffold-grouped, molecules sharing a scaffold never straddle folds, 5 seeds |")
w("| Cliff definitions | `cliffA` (dataset-level: participates in a pair with `sim ≥ 0.85` and `\\|ΔpKi\\| ≥ 1.0`), `cliffB` (**mechanism**, the higher-activity side of such a pair where the pair is established through *training* neighbours), `strong cliff` (`\\|ΔpKi\\| ≥ 2.0`) |")
w()
w("[saved]")
w("---")
w()

# ═══════════════════════ S2 ═══════════════════════
A = pd.read_csv(os.path.join(C, "results_sens", "SENSITIVITY.csv"))
CFG_ORDER = ["base", "sim075", "sim095", "act050", "act150", "bins03", "bins08"]
M_ORDER = ["mondrian_wstd", "mondrian_std", "mondrian_iqr", "mondrian_nnsim",
           "mondrian_pairgap", "mondrian_pair", "mondrian_nb", "mondrian_rfstd"]
w("## S2  Threshold sensitivity — all 7 configurations")
w()
w("Single-factor sweeps around the main configuration (`sim = 0.85`, `act = 1.0`, `n_bins = 5`): "
  "`sim ∈ {0.75, 0.85, 0.95}`, `act ∈ {0.5, 1.0, 1.5}`, `n_bins ∈ {3, 5, 8}` — 7 distinct configurations "
  "in total. Each cell reports the mean change in **cliff-group coverage** relative to global (pp), as the "
  "within-dataset mean over seeds followed by a paired Wilcoxon test over datasets.")
w()

def piv(col, fmt):
    t = A[A.method.isin(M_ORDER)].pivot_table(index="method", columns="config", values=col)
    t = t.reindex(index=M_ORDER, columns=[c for c in CFG_ORDER if c in t.columns])
    return t

w("### S2.1  Δ cliff-group coverage (pp)")
w()
w(md_table(piv("delta_mean", "%.4g").map(lambda v: "" if pd.isna(v) else "%+.2f" % (v * 100)), index=True))
w()
w("### S2.2  Paired Wilcoxon p-value")
w()
w(md_table(piv("wilcoxon_p", "%.3g").map(lambda v: "" if pd.isna(v) else "%.3g" % v), index=True))
w()
w("### S2.3  Direction consistency (datasets with Δ>0 / datasets)")
w()
_p = piv("n_ds_delta_pos", "%d"); _n = piv("n_ds", "%d")
comb = (_p.astype("Int64").astype(str) + "/" + _n.astype("Int64").astype(str)).replace({"<NA>/<NA>": "—"})
w(md_table(comb, index=True))
w()
w("### S2.4  Width cost (relative to global)")
w()
w(md_table(piv("width_delta", "%.4g").map(lambda v: "" if pd.isna(v) else "%+.2f%%" % (v * 100)), index=True))
w()
w("**Reading.** `d_wstd` is positive in **all 7 configurations** (significantly in 3: `act=1.5`, `bins=8`, "
  "`sim=0.75`) with direction consistency 8/15–12/15 and a width cost of +0.95% to +7.54%. `d_nb` is "
  "**significantly negative in all 6 usable configurations** (p ≤ 0.020) with direction consistency ≤5/15. "
  "`rf_std` is significantly negative in 5 of 7 but **reverses at `act=1.5` (+6.2 pp, p = 0.003)** — a "
  "reversal listed as a limitation of the main text (§4.3). The `sim=0.95` configuration is **unusable**: "
  "cliff pairs become so sparse that only 1 of 15 datasets survives, and its numbers are not interpreted.")
w()
w("---")
w()

# ═══════════════════════ S3 ═══════════════════════
ADS = pd.read_csv(os.path.join(C, "results_ecfp_abl", "AD_SENSITIVITY.csv"))
w("## S3  Applicability domain — all 8 definitions")
w()
w("Two AD proxy families × 4 quantile thresholds (10/20/30/40%). All cross-dataset comparisons use the "
  "**within-group rank quantile** of the proxy, so that large datasets do not dominate. "
  "\u201cIn-domain\u201d means the within-group rank quantile is at or above the threshold.")
w()
w("### S3.1  Per-definition summary")
w()
t = ADS.copy()
t["family"] = t["proxy"]
t = t[["family", "ad_quantile", "n_in_ad", "share_of_failures_in_ad", "fail_rate_in_ad",
       "fail_rate_out_ad", "cov_in_ad_cliff", "fail_in_ad_cliff", "n_in_ad_cliff",
       "cov_in_ad_noncliff", "n_cliff_out_ad", "in_ad_recall", "in_ad_recall_lift"]]
t.columns = ["AD proxy", "threshold", "n in AD", "share of failures in AD (%)", "fail rate in AD",
             "fail rate out of AD", "coverage of in-AD cliffs", "fail rate of in-AD cliffs",
             "n in-AD cliffs", "coverage of in-AD non-cliffs", "n cliffs out of AD",
             "in-AD recall (20% budget)", "recall lift vs random"]
w(md_table(t))
w()
w("**Reading.** 40.8–75.9% of all conformal failures occur *inside* the domain, and in-domain cliffs fail "
  "at 26.0–28.5% versus 6.3–7.3% for in-domain non-cliffs — a **3.4–5.3× ratio in all 8 definitions**. The "
  "distance-type AD contains **no** cliff outside the domain (0 in every configuration); the density-type AD "
  "leaves only 23–607 molecules (0.03–0.87%) outside, so >99.1% of all cliffs are in-domain under either family.")
w()
w("### S3.2  The 2×2 decomposition (threshold = 20%, distance-type AD)")
w()
A22 = pd.read_csv(os.path.join(C, "results_ecfp_abl", "AD_2x2.csv"))
A22.columns = ["group", "n", "share of test set (%)", "coverage", "failure rate", "share of failures (%)"]
w(md_table(A22))
w()
w("Only three cells are populated: **the distance-type AD contains no cliff outside the domain**, which is "
  "itself the point — cliffs are not outliers in chemical space, they are the *dense* region.")
w()
w("### S3.3  Coverage and failure rate by similarity bin")
w()
B = pd.read_csv(os.path.join(C, "results_ecfp_abl", "AD_BY_SIMBIN.csv"))
B.columns = ["similarity bin (0 = sparsest)", "n", "share of test set (%)", "coverage",
             "failure rate", "share of failures (%)", "cliff rate (%)"]
w(md_table(B))
w()
w("The cliff rate rises monotonically with similarity (0% in the two sparsest bins → 16.8% in the densest), "
  "which is the direct counterexample to \u201cfar from the training set ⇒ unreliable\u201d: the failures that matter "
  "sit at the *centre* of the domain.")
w()
w("### S3.4  Recall curves and the missing-neighbour convention")
w()
R = pd.read_csv(os.path.join(C, "results_ecfp_abl", "AD_RECALL_CURVES.csv"))
w(md_table(R))
w()
w("**Denominator warning.** `failure_recall` above is the recall over **all** failures "
  "(weighted across datasets by failure counts), at the given flagging budget, using "
  "within-group rank quantiles. The `in_ad_recall` column in S3.1 uses a **different "
  "denominator** \u2014 failures **inside** the AD only. The two are reported separately and "
  "must never be merged into one panel or one comparison; at a 20% budget the AD rule "
  "(flagging the sparsest) recalls 38.2% of all failures versus 30.4% for the dispersion "
  "rule, so the AD rule is itself a strong triage baseline and the dispersion rule's value "
  "lies in complementarity (union 41.2%), not in a higher total.")
w()
N = pd.read_csv(os.path.join(C, "results_ecfp_abl", "NO_NEIGHBOR_CONVENTION.csv"))
N.columns = ["metric"] + list(N.columns[1:])
w(md_table(N.rename(columns={N.columns[0]: "quantity", N.columns[1]: "NaN → sparsest bin (default)",
                             N.columns[2]: "drop molecule"}), index=False))
w()
w("All four key quantities agree in direction under both conventions, and the `drop` variant is slightly "
  "stronger (share of failures in-domain 61.8% → 66.6%; recall 0.368 → 0.378).")
w()
w("---")
w()

# ═══════════════════════ S4 ═══════════════════════
BT = pd.read_csv(os.path.join(C, "results_ecfp_abl", "COVERAGE_BACKTEST.csv"))
w("## S4  Coverage backtests (correct statistical framing)")
w()
w("The conformal guarantee is **marginal over the randomness of calibration + test**, and test molecules "
  "within one split share a model and are positively correlated. The binomial distribution is therefore **not** "
  "the correct null within a single split. The formal test used here is a **split-level z-test**: the statistic "
  "is the mean within-split coverage over the 150 (dataset, split) groups, tested against `H0: mean = 0.90`. "
  "Kupiec POF and Christoffersen conditional coverage are reported as references.")
w()
t = BT.copy()
t.columns = ["method", "marginal coverage", "nominal", "n test", "n fail", "pooled Kupiec p",
             "n split groups", "within-split SD", "p5", "p95", "split-level z", "split-level p",
             "Kupiec reject @5% (n)", "Kupiec reject rate", "Christoffersen reject @5% (n)",
             "Christoffersen reject rate", "median Kupiec p", "median Christoffersen p"]
w(md_table(t))
w()
w("Every method attains marginal coverage within 0.899–0.903. The global split-level z-test gives "
  "p = 0.54. Some Mondrian variants over-cover very slightly (`d_nb` 0.9026, split-level p = 0.016) — a real, "
  "if small, cost of conditionalisation. Christoffersen shows no clustering (median p ≈ 0.24–0.36). The "
  "elevated within-split Kupiec rejection rate (~12–22%) is **expected** given the correlation between "
  "molecules in a split and is not evidence against validity.")
w()
w("---")
w()

# ═══════════════════════ S5 ═══════════════════════
H = pd.read_csv(os.path.join(C, "results_phase14", "TRANSFER_HARM.csv"))
agg = H.groupby(["run", "signal"]).mean(numeric_only=True).reset_index()
spear = agg["net_mag"].corr(agg["d_cov_cliff"], method="spearman")
pear = agg["net_mag"].corr(agg["d_cov_cliff"])
sp_t = agg["net_tight"].corr(agg["d_cov_cliff"], method="spearman")
pe_t = agg["net_tight"].corr(agg["d_cov_cliff"])
sig = H.groupby("signal").mean(numeric_only=True).reset_index()
order = ["d_wstd", "d_std", "d_nn_sim", "d_iqr", "d_pair", "d_pair_gap", "rf_std", "d_nb"]
sig = sig.set_index("signal").reindex(order).reset_index()

w("## S5  Proposition 1: proof and empirical validation")
w()
w("### S5.1  Statement and proof")
w()
w("**Proposition 1 (coverage-transfer identity).** Let `ε_i = \\|y_i − ŷ_i\\|` be the nonconformity score of "
  "test molecule `i`, `q̄` the global conformal quantile, and `q_{k(i)}` the quantile of the bin that signal "
  "`s` assigns to `i`. Then for **any** subgroup `G`:")
w()
w("```")
w("ΔCov_G := Cov_G(Mondrian_s) − Cov_G(global)")
w("        = (1/|G|) · [ #{i ∈ G : q̄ < ε_i ≤ q_{k(i)}} − #{i ∈ G : q_{k(i)} < ε_i ≤ q̄} ]")
w("```")
w()
w("**Proof.** Coverage of `i` is `1{ε_i ≤ q}`. For a single `i`,")
w()
w("```")
w("1{ε_i ≤ q_{k(i)}} − 1{ε_i ≤ q̄}")
w("```")
w()
w("takes the value `+1` exactly when `q̄ < ε_i ≤ q_{k(i)}` (the interval was widened and a failure was "
  "recovered), takes `−1` exactly when `q_{k(i)} < ε_i ≤ q̄` (the interval was tightened and a failure was "
  "newly created), and is `0` otherwise — in particular when both quantiles lie on the same side of `ε_i`. "
  "Summing over `i ∈ G` and dividing by `|G|` gives the identity. ∎")
w()
w("The identity is exact, finite-sample, and distribution-free: it holds pathwise for every split and every "
  "realisation of the calibration set, with no exchangeability or asymptotic argument. Effectively the same "
  "point can be made for a *population* level statement via a conservation law for pooled calibration; "
  "Eq. (S5.1) is the finite-sample accounting version, which is what makes per-subgroup attribution possible.")
w()
w("**Corollary (harm criterion).** `s` harms `G` if and only if `G`'s failures fall disproportionately into "
  "the **tightened** bins — that is, if the signal deems `G`'s region \u201ceasy\u201d (low-score bins) while failing to "
  "predict its failures. This is precisely the mechanism behind `M(s) > 0` in the main text.")
w()
w("### S5.2  The label-free diagnostic")
w()
w("Tightening occurs in bins with small calibration scores, so it suffices to quantify how much the quantile "
  "assigned to `G` moves relative to the global one. Define the **net tightening magnitude**")
w()
w("```")
w("net_mag(G; s) = E_{i∈G}[ max(0, (q_{k(i)} − q̄)/q̄) ] − E_{i∈G}[ max(0, (q̄ − q_{k(i)})/q̄) ]")
w("```")
w()
w("Positive values mean `G` is net-widened (it should benefit); negative values mean it is net-tightened "
  "(it should be harmed). Both `q_{k(i)}` and `q̄` depend only on the **calibration** set, and `G` on the "
  "training set and its labels, so `net_mag` **requires no test labels** and is computable before deployment. "
  "The companion quantity `net_tight` (the difference in the *fractions* of `G` that are tightened and widened) "
  "is reported for comparison; it fails to flag `rf_std`, which tightens few molecules but by a large amount.")
w()
w("### S5.3  Validation")
w()
w(f"**Identity.** Across 240 (dataset × signal) groups, the maximum absolute residual between the two sides "
  f"of Eq. (S5.1) is **1.04×10⁻¹⁶** (median 3.30×10⁻¹⁷).")
w()
w(f"**Diagnostic.** Aggregating to n = 16 (8 signals × 2 representations): "
  f"Spearman(`net_mag`, observed ΔCov) = **{spear:+.3f}** (Pearson {pear:+.3f}); "
  f"Spearman(`net_tight`, ΔCov) = {sp_t:+.3f} (Pearson {pe_t:+.3f}).")
w()
_stat_path = os.path.join(C, "results_phase14", "STAT_STRENGTH.txt")
if os.path.exists(_stat_path):
    w("**Statistical strength of this correlation** (produced by "
      "`scripts/phase14_stat_strength.py`; the 16 units are not independent, so both a unit-level "
      "and a signal-level test are reported):")
    w()
    w("```")
    for _ln in io.open(_stat_path, encoding="utf-8").read().splitlines():
        if _ln.startswith("==="):
            continue
        w(_ln)
    w("```")
    w()
    w("Unit-level figures use 200,000 permutations and 10,000 bootstrap resamples; the signal-level "
      "test (n = 8) removes the correlation induced by the shared representation axis, and remains "
      "significant at p = 0.0027. Leave-one-out moves ρ only within [0.843, 0.889], so no single "
      "unit carries the result.")
    w()
tt = sig[["signal", "net_mag", "net_tight", "d_cov_cliff"]].copy()
tt.columns = ["signal", "net_mag (predicted)", "net_tight (predicted)", "observed ΔCov (fraction)"]
tt["observed ΔCov (pp)"] = tt["observed ΔCov (fraction)"].map(lambda v: "%+.2f" % (v * 100))
tt = tt.drop(columns=["observed ΔCov (fraction)"])
w(md_table(tt))
w()
w("Signals with positive `net_mag` (`d_wstd` +0.063, `d_std` +0.060, `d_nn_sim` +0.037, `d_iqr` +0.033) are "
  "indeed the ones that do not harm the cliff group, and the two signals with negative `net_mag` (`rf_std` "
  "−0.016, `d_nb` −0.046) are precisely the harmful ones. Note that `net_tight` **fails** on `rf_std` "
  "(+0.135, i.e. it predicts *no* harm) because `rf_std` tightens relatively few molecules but by a large "
  "amount — the magnitude-weighted version is required.")
w()
w("---")
w()
w("### S5.4  Direct test: why `d_nn_sim` (largest misalignment) causes no harm")
w()
w("A ground-truth dump from **inside** the pipeline (`phase11_shift.py --dump-signals`; "
  "analysis script `scripts/phase16_mechanism.py`, output `results_server/PHASE16_MECHANISM.txt`; "
  "15 datasets x random split x 3 seeds) makes the mechanism of the `d_nn_sim` counterexample "
  "directly measurable. Harm is governed by the **calibration conformal quantile of the bin a "
  "cliff molecule falls into, relative to the global quantile** (bin-quantile ratio, q-ratio):")
w()
w("| Signal | q-ratio (cliff bin) | rebuilt ΔCov_cliff (pp) | known measured |")
w("|---|---|---|---|")
w("| `d_wstd` | 1.055 | +0.23 | +0.36 pp (n.s.) |")
w("| `d_std` | 1.056 | −0.40 | +1.4 pp |")
w("| `d_iqr` | 1.068 | +2.46 | ≈ 0 |")
w("| `d_nb` | **0.962** | **−1.51** | **−2.70 pp (p = 6×10⁻³)** |")
w("| `d_nn_sim` | 1.045 | +3.22 | +1.7 pp |")
w()
w("`d_nb` is the **only** signal whose cliff bin quantile sits below the global quantile — 60% of "
  "cliff molecules are tightened, producing the observed harm. All other signals place cliff "
  "molecules in bins whose quantile is 4.5–6.8% *above* global, so no harm occurs.")
w()
w("**Resolution of the counterexample.** The harmlessness of `d_nn_sim` is an artifact of the "
  "binning convention: `np.digitize` sends NaN (no-neighbour) molecules to the **top** bin, and "
  "no-neighbour molecules are precisely the hard ones (mean residual 1.85× the rest; 321/321 "
  "test molecules land in the top bin). Their large residuals inflate the top bin's quantile, "
  "which accidentally *protects* the cliff molecules sharing that bin. Under a naive NaN→0 "
  "convention the same signal would show ≈ −12 pp of harm. The `d_nn_sim` cell should therefore "
  "be read as evidence of **implementation-convention sensitivity**, not as a failure of the "
  "misalignment criterion. Implementation validation: reconstructed bins agree with the "
  "pipeline’s ground-truth bins on 85–87% of molecules (`d_wstd` 85.4%, `d_nb` 87.0%).")
w()
w("---")
w()

# ═══════════════════════ S6 ═══════════════════════
w("## S6  Split realism and the signal-precision ladder")
w()
w("### S6.1  R1 — the cliff gap survives scaffold splitting")
w()
w("| Split · cliff definition | Δ (non-cliff − cliff) | direction consistency | paired Wilcoxon |")
w("|---|---|---|---|")
w("| random · `cliffA` (dataset-level, primary read-out) | +9.5 pp | 14/15 | p = 6.1×10⁻⁵ |")
w("| random · `cliffB` (mechanism) | +17.3 pp | 15/15 | p = 3.1×10⁻⁵ |")
w("| **scaffold · `cliffA`** | **+5.0 pp** | **13/15** | **p = 4.3×10⁻⁴** |")
w("| scaffold · `cliffB` | +8.9 pp | 12/15 | p = 2.1×10⁻³ |")
w()
w("Marginal coverage: 0.8997 under random splitting (per-dataset 0.882–0.932) and 0.8964 under scaffold "
  "splitting (0.865–0.925), with 4 of 15 datasets below 0.88 under scaffold splitting. `cliffB` becomes "
  "sparse under scaffold splitting (test-set share 4.05% → 2.47%) because same-scaffold neighbours leave the "
  "training set; the split-invariant `cliffA` is therefore used as the primary read-out.")
w()
w("### S6.2  R2 — additional baselines (8 methods × 2 splits × 5 seeds)")
w()
w("| Method | random Δ (pp) | consistency | scaffold Δ (pp) | consistency | width Δ (random / scaffold) |")
w("|---|---|---|---|---|---|")
w("| global (reference) | — | — | — | — | — |")
w("| CQR | +2.73 (p=5.4×10⁻³) | 13/15 | +1.07 (p=0.25) | 10/15 | +6.3% / +2.9% |")
w("| adaptive (normalised) | +2.54 (p=0.055) | 12/15 | −0.34 (p=0.98) | 9/15 | +21.0% / +27.9% |")
w("| cluster | +1.15 (p=0.041) | 12/15 | +0.26 (p=0.68) | 10/15 | +4.2% / +1.8% |")
w("| weighted (covariate shift) | −1.26 (p=0.036) | 4/15 | −1.94 (p=0.016) | 3/15 | −1.4% / −2.6% |")
w("| kNN-weighted | −1.31 (p=0.083) | 5/15 | −1.67 (p=0.048) | 4/15 | −3.2% / −3.1% |")
w("| Mondrian ← `rf_std` | −3.96 (p=4.3×10⁻⁴) | 2/15 | −5.28 (p=3.1×10⁻⁴) | 1/15 | −1.3% / −0.6% |")
w("| Mondrian ← `d_wstd` (aligned) | +0.36 (p=0.36) | 9/15 | −0.71 (p=0.19) | 5/15 | +2.4% / +1.4% |")
w("| Mondrian ← `d_nb` (misaligned) | −2.70 (p=6.3×10⁻³) | 3/15 | −3.82 (p=1.5×10⁻³) | 2/15 | +2.6% / +1.8% |")
w()
w("The aligned signal never harms under either split; both misaligned signals are significantly harmful under "
  "both. Adaptive CP improves the cliff group under random splitting (+2.54 pp; mechanism-definition `cliffB` "
  "+6.37 pp, 13/15, p = 3.4×10⁻³) but at a **+21% width cost** and with the gain vanishing under scaffold "
  "splitting. Covariate-shift weighted CP, which is designed for a shift type not present here, makes the "
  "cliff group *worse* under both splits.")
w()
w("### S6.3  R3 — the signal-precision ladder (frozen predictor)")
w()
w("The predictor is frozen (the mean of the full 200-tree forest, constant throughout) and only the precision "
  "of the uncertainty signal is varied: the SD over random `k`-tree subsets, `k ∈ {2, 5, 10, 25, 50, 100, 200}`. "
  "Predictions, partitions and residuals are unchanged, so the signal's information content is the sole variable.")
w()
w("| k | AUC_fail | AUC_cliffB | misalignment M | Δ cliff coverage `cliffA` (pp) | Δ cliff coverage `cliffB` (pp) |")
w("|---|---|---|---|---|---|")
for _, r in pd.DataFrame([
        (2, .592, .493, +.099, -0.37, -0.57), (5, .647, .539, +.108, -0.45, -0.87),
        (10, .678, .545, +.132, -2.39, -1.38), (25, .702, .524, +.178, -3.03, -3.34),
        (50, .712, .529, +.182, -3.86, -4.68), (100, .719, .533, +.186, -3.91, -3.59),
        (200, .722, .532, +.190, -3.96, -4.38)],
        columns=["k", "a", "b", "m", "ca", "cb"]).iterrows():
    w(f"| {int(r.k)} | {r.a:.3f} | {r.b:.3f} | {r.m:+.3f} | {r.ca:+.2f} | {r.cb:+.2f} |")
w()
w("AUC_fail rises strictly monotonically with `k` (Spearman ρ = +1.000 on k-aggregated means, +0.618 "
  "row-wise, p = 1.1×10⁻⁵⁶) while AUC_cliffB barely moves, so the misalignment measure rises monotonically. "
  "The harmful effect appears and grows: |ΔCov| correlates with `k` at +0.893 (k-aggregated); the dose effect "
  "ΔCov(k=100,200) − ΔCov(k=2,5) is −3.26 pp (Wilcoxon p = 3.2×10⁻⁴; 45/75 pairs negative); the most "
  "informative signal, k = 200, still harms by −4.38 pp (p = 1.0×10⁻², 12/15). This supplies the controlled "
  "test that the member-count experiment of the main text (§3.6) cannot provide, because there the model and "
  "the signal changed together.")
w()
w("### S6.4  Reproducibility cross-check")
w()
w("After the `NaN`-binning fix described in S1.3, the affected arms were re-run on a reduced method set. "
  "The re-run reproduces the first run exactly on the one arm common to both: `Mondrian ← rf_std` coverage is "
  "identical to the last decimal across both runs on the same 150 (split, dataset, seed) triplets "
  "(maximum absolute difference **0.000**), so the re-run numbers can be spliced into the same table as the "
  "first-run numbers.")
w()
w("---")
w()

# ═══════════════════════ S7 ═══════════════════════
w("## S7  Pre-registered analysis plan (verbatim)")
w()
w("The plan below was written before the confirmatory analyses were run. It fixes the primary/secondary/"
  "exploratory split, the Holm–Bonferroni correction within the primary family, the direction-consistency "
  "criterion, the disclosure obligations for counterexamples, and a table of every change made outside the "
  "plan. Section 6 of the plan records the S6–S8 verdicts and the one case in which a pre-registered "
  "expectation was contradicted by the data.")
w()
pre_path = os.path.join(C, "预注册_分析计划_EN.md")
if not os.path.exists(pre_path):
    pre_path = os.path.join(C, "预注册_分析计划.md")
pre = io.open(pre_path, encoding="utf-8").read()
# 降两级，使预注册的标题嵌在 S7 之下，不打乱 SI 的章节层级
pre = re.sub(r"^(#{1,5}) ", lambda m: "#" * min(len(m.group(1)) + 2, 6) + " ", pre, flags=re.M)
w(pre)
w()
w("---")
w()

# ═══════════════════════ S8 ═══════════════════════
CM = pd.read_csv(os.path.join(C, "results_ecfp_abl", "CQR_COVERAGE_MATCHED.csv"))
w("## S8  Method ladder at matched marginal coverage (including CQR)")
w()
w("Each method run at `α ∈ {0.07, 0.10, 0.13, 0.16}` (each a genuinely usable conformal method, no post-hoc "
  "rescaling), interpolated per (dataset, split) to a marginal coverage of 0.90, and read off there. "
  "Cliff share = 4.08%.")
w()
t = CM.copy()
t.columns = ["method", "note", "α for target", "marginal at", "cliff-group coverage (`cliffB`)",
             "cliff-group coverage (`cliffA`)", "mean width"]
w(md_table(t.drop(columns=["note"])))
w()
gap = (t["marginal at"] - t["cliff-group coverage (`cliffB`)"]) * 100
tt = pd.DataFrame({"method": t["method"], "cliff-group coverage": t["cliff-group coverage (`cliffB`)"],
                   "gap (pp)": gap.map(lambda v: "%.1f" % v),
                   "width": t["mean width"].map(lambda v: "%.4f" % v),
                   "width vs global": (((t["mean width"] / t.loc[t.method == "global", "mean width"].iloc[0]) - 1) * 100).map(lambda v: "%+.1f%%" % v)})
w(md_table(tt))
w()
w("Ordering by cliff-group coverage: **CQR (0.817) > `d_wstd` (0.741) > `d_std` (0.737) > global (0.729) > "
  "`d_nn_sim` > `d_iqr` > `d_pair` > `d_pair_gap` > `rf_std` (0.677) > `d_nb` (0.671)**. The ordering matches "
  "the misalignment prediction, and CQR is the strongest repair (gap 8.6 pp) but costs +6.3% in width.")
w()
w("---")
w()

# ═══════════════════════ S9 ═══════════════════════
OUT.append("## S9  Alpha sensitivity (raw-coverage view)")
OUT.append("")
OUT.append("The cliff coverage gap is not an artifact of alpha = 0.10. Re-analysing the archived alpha "
  "sweep (phase1_signal, 4 seeds x 15 datasets, no CQR) at the raw-coverage view:")
OUT.append("")
OUT.append("| alpha | global gap (pp) | Mondrian<-d_wstd | Mondrian<-d_nb | Mondrian<-rf_std | marginal cov (global) |")
OUT.append("|---|---|---|---|---|---|")
OUT.append("| 0.07 | +13.2 (14/15, p=1e-4) | +12.1 (15/15) | +16.6 (14/15) | +18.5 (15/15) | 0.932 |")
OUT.append("| 0.10 | +17.9 (15/15) | +16.2 (15/15) | +23.0 (15/15) | +22.9 (15/15) | 0.900 |")
OUT.append("| 0.13 | +22.5 (15/15) | +19.5 (15/15) | +28.2 (15/15) | +26.2 (15/15) | 0.872 |")
OUT.append("| 0.16 | +25.8 (15/15) | +22.5 (15/15) | +31.1 (15/15) | +29.8 (15/15) | 0.843 |")
OUT.append("| 0.20 | +30.1 (15/15) | +25.4 (15/15) | +34.2 (15/15) | +31.7 (15/15) | 0.803 |")
OUT.append("")
OUT.append("Under the split-invariant cliffA definition the gap grows from +7.3 to +17.9 pp. The gap "
  "**grows monotonically with alpha** while marginal coverage tracks the nominal level: cliff "
  "failures sit in the heavy tail of the residual distribution, so wider intervals do not rescue "
  "them. (Analysis script `scripts/phase17_alpha_sensitivity.py`; archived results "
  "`results/alpha_007..020`.)")
OUT.append("")
OUT.append("---")
OUT.append("")

# ═══════════════════════ S10 ═══════════════════════
OUT.append("## S10  Extrapolation to classification (exploratory)")
OUT.append("")
OUT.append("Set-valued conformal prediction (alpha = 0.10, ECFP4 + 200-tree random forest, random "
  "60/20/20 splits, 10 seeds) on five MoleculeNet benchmarks. The classification analogue of an "
  "activity cliff is a **label cliff**: a test molecule with a training neighbour "
  "(Tanimoto >= 0.85) carrying a different label.")
OUT.append("")
OUT.append("| Dataset | n | cliffs / seed (max) | cov (global) | cliff | non-cliff | gap (pp) | direction | APS cliff (size) | Mondrian<-nnsim cliff |")
OUT.append("|---|---|---|---|---|---|---|---|---|---|")
OUT.append("| BBBP | 1,975 | ~4 | 0.893 | 0.417 | 0.897 | +48.0 | 7/10 | 0.967 (2.00) | 0.417 |")
OUT.append("| BACE | 1,513 | ~27 | 0.905 | 0.573 | 0.931 | +35.7 | 10/10 | 0.996 (1.99) | 0.581 |")
OUT.append("| ClinTox | 1,461 | ~3 | 0.909 | 0.429 | 0.912 | +48.9 | 5/10 | 1.000 (2.00) | 0.429 |")
OUT.append("| HIV | 41,120 | ~41 | 0.904 | 0.040 | 0.908 | +86.8 | 10/10 | 0.997 (2.00) | 0.067 |")
OUT.append("| Tox21_NR-AR | 7,258 | ~10 | 0.903 | 0.027 | 0.907 | +88.0 | 10/10 | 1.000 (2.00) | 0.081 |")
OUT.append("")
OUT.append("Direction is consistent in 10/10 seeds for BACE, HIV and Tox21-NR-AR, 7/10 for BBBP and "
  "5/10 for ClinTox (label-cliff groups average only 1.5\u201337 molecules per seed, and some "
  "ClinTox seeds contain no cliff molecule, so the smaller-seed counts reflect trend rather than "
  "high-powered conclusions). Class-conditional conditioning on the predicted class coincides "
  "with global on all five datasets (gap identical to the column above): in binary problems, "
  "per-class quantiles almost coincide with the global quantile. APS reaches \u22481.0 coverage "
  "only by expanding the set to both labels (size \u2248 2.0), i.e. a degenerate solution that "
  "forfeits discriminative power. Similarity conditioning (mondrian_nnsim) improves label-cliff "
  "coverage by at most +5.4 pp, insufficient to close the gap. (Script "
  "`scripts/phase17_classify.py`; results `results/phase17_classify_v2/`.)")
OUT.append("")
OUT.append("---")
OUT.append("")

# ═══════════════════════ S5.5 ═══════════════════════
OUT.append("### S5.5  Merging the two criteria: `net_mag` (magnitude) x bin-quantile ratio (mechanism)")
OUT.append("")
OUT.append("Both criteria are computable **before deployment** (calibration residuals + signal + subgroup "
  "definition only; no test labels). Analysis script `scripts/phase19_merged_criterion.py`, output "
  "`results_server/S55_MERGED_CRITERION.txt`.")
OUT.append("")
OUT.append("Signal level (two representations pooled, n = 8):")
OUT.append("")
OUT.append("| Signal | `net_mag` (pp) | observed \u0394Cov (pp) | q-ratio | `net_mag` sign correct |")
OUT.append("|---|---|---|---|---|")
OUT.append("| `d_wstd` | +6.30 | +1.74 | 1.061 | \u2713 |")
OUT.append("| `d_std` | +5.97 | +1.85 | 1.056 | \u2713 |")
OUT.append("| `d_iqr` | +3.30 | +0.54 | 1.068 | \u2713 |")
OUT.append("| `d_nn_sim` | +3.68 | +1.66 | 1.045 | \u2713 |")
OUT.append("| `d_pair` | \u22120.40 | \u22120.27 | n/a | \u2713 |")
OUT.append("| `d_pair_gap` | \u22120.38 | \u22120.62 | n/a | \u2713 |")
OUT.append("| `rf_std` | \u22121.63 | \u22125.85 | n/a | \u2713 |")
OUT.append("| `d_nb` | \u22124.64 | \u22123.62 | 0.962 | \u2713 |")
OUT.append("")
OUT.append("**Sign agreement: `net_mag` 8/8 at signal level; 11/13 at unit level** (the two dissenting "
  "units both have |`net_mag`| < 0.4 pp, inside the declared *no-substantive-effect* zone); "
  "**q-ratio 5/5** on the five signals with ground-truth bins. Deployable reading: `net_mag` gives "
  "the *magnitude* (|net_mag| < 0.5 pp -> no substantive effect; < 0 -> warn), the bin-quantile "
  "ratio gives the *mechanism*. Known residual weakness: `rf_std`'s harm comes from a few "
  "large-magnitude tightenings with little calibration-side mass, which `net_mag` underestimates "
  "(the bin-quantile ratio is immune, since it looks directly at the bins where cliff molecules "
  "land) \u2014 the two criteria together produce **no missed warnings** on the available data.")
OUT.append("")
OUT.append("---")
OUT.append("")
OUT.append("*End of Supporting Information.*")

io.open(os.path.join(C, "SI_v1.md"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")
print("SI_v1.md written | sections:", sum(1 for x in OUT if x.startswith("## ")), "| chars:", sum(len(x) + 1 for x in OUT))
