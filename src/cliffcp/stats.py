"""统计检验与分组覆盖评估。

两个关键设计（决定这篇能不能过审）：

1. 覆盖率必须报区间，不能只报一个数。
   几百条测试集上，经验覆盖率本身的抽样误差就能到 ±3 个点，
   只报点值会让"差异"落在噪声里。

2. 组间比较必须控制活性量程（potency bin）。
   Scientific Reports 2024 (s41598-024-57135-6) 已经证明：
   化合物效力预测的校准质量随效力区间变化（高效力段过度自信、中间段过度不自信）。
   如果不控制这一项，"悬崖组覆盖更低"可能只是"悬崖分子恰好落在某个效力段"。
   所以主检验用**分层置换检验**：在每个效力分箱内置换组标签，
   得到一个已扣除效力效应的零分布。
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    """二项比例的 Wilson 区间（比正态近似稳，小样本/极端比例下不越界）。"""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def coverage_stats(covered: np.ndarray):
    """给定 0/1 覆盖指示向量，返回覆盖率 + Wilson 区间 + n。"""
    c = np.asarray(covered).astype(bool)
    n = int(c.size)
    k = int(c.sum())
    lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "k": k,
        "coverage": (k / n) if n else float("nan"),
        "ci_lo": lo,
        "ci_hi": hi,
    }


def potency_bins(y: np.ndarray, n_bins: int = 5):
    """按活性值的分位数分箱；重复边界自动减少箱数。"""
    y = np.asarray(y, float)
    try:
        edges = np.unique(np.quantile(y, np.linspace(0, 1, n_bins + 1)))
    except Exception:
        return np.zeros(y.size, dtype=np.int64)
    if edges.size < 3:
        return np.zeros(y.size, dtype=np.int64)
    return np.clip(np.digitize(y, edges[1:-1], right=False), 0, edges.size - 2)


def stratified_permutation_test(
    covered: np.ndarray,
    group: np.ndarray,
    strata: np.ndarray,
    n_perm: int = 4000,
    seed: int = 0,
):
    """分层置换检验。

    统计量 = coverage(非悬崖) - coverage(悬崖)。正值 = 悬崖组被欠覆盖。

    做法：在每个效力分箱内独立置换组标签，保持各箱的组比例不变，
    从而把"效力量程"这个混杂因子扣掉。
    """
    covered = np.asarray(covered).astype(np.float64)
    group = np.asarray(group).astype(bool)
    strata = np.asarray(strata)

    if group.sum() == 0 or (~group).sum() == 0:
        return {"stat": float("nan"), "p_value": float("nan"), "n_perm": 0}

    def stat_of(gmask):
        a = covered[~gmask]
        b = covered[gmask]
        if a.size == 0 or b.size == 0:
            return np.nan
        return float(a.mean() - b.mean())

    obs = stat_of(group)

    rng = np.random.default_rng(seed)
    idx_by_stratum = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    null = np.empty(n_perm, dtype=np.float64)
    for r in range(n_perm):
        perm = group.copy()
        for idx in idx_by_stratum:
            if idx.size > 1:
                perm[idx] = group[rng.permutation(idx)]
        null[r] = stat_of(perm)

    null = null[np.isfinite(null)]
    if null.size == 0:
        return {"stat": float(obs), "p_value": float("nan"), "n_perm": 0}
    # 单侧：悬崖组覆盖更低（gap>0）
    p = float((np.sum(null >= obs) + 1) / (null.size + 1))
    return {
        "stat": float(obs),
        "p_value": p,
        "n_perm": int(null.size),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
    }


def one_sample_vs_nominal(covered: np.ndarray, alpha: float, side: str = "less"):
    """检验组内覆盖率是否显著低于名义值 1-alpha。"""
    c = np.asarray(covered).astype(bool)
    n = int(c.size)
    k = int(c.sum())
    if n == 0:
        return {"p_value": float("nan"), "n": 0}
    res = stats.binomtest(k, n, 1.0 - alpha, alternative=side)
    return {"p_value": float(res.pvalue), "n": n, "k": k}


def paired_wilcoxon(a: np.ndarray, b: np.ndarray):
    """匹配样本比较（例如各随机划分下 方法X 与 方法Y 的覆盖率）。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 6 or np.allclose(a[m], b[m]):
        return {"p_value": float("nan"), "n": int(m.sum())}
    res = stats.wilcoxon(a[m], b[m])
    return {"p_value": float(res.pvalue), "n": int(m.sum())}


# ---------------------------------------------------------------- 信号评估


def auc_detect(signal: np.ndarray, bad: np.ndarray) -> float:
    """信号对"坏结果"的判别 AUC。

    为什么用 AUC 而不是设阈值分组：阈值是任意的，换一个阈值结论就可能变。
    AUC 是阈值无关的整体判别力度量，审稿人无法用"你挑的阈值"来反驳。

    signal 越大应越倾向于 bad（约定：信号是"不可靠程度"）。
    """
    s = np.asarray(signal, float)
    b = np.asarray(bad).astype(bool)
    m = np.isfinite(s)
    s, b = s[m], b[m]
    if b.sum() == 0 or (~b).sum() == 0 or s.size < 10:
        return float("nan")
    # Mann–Whitney U 的等价形式，带并列平均秩
    r = stats.rankdata(s)
    n1 = int(b.sum())
    n0 = int((~b).sum())
    r1 = r[b].sum()
    auc = (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)


def bin_edges_from(reference: np.ndarray, n_bins: int = 5):
    """从参考分布（建议用训练集/全局信号，而非校准集）取分箱边界。

    这样做是为了让分箱规则不依赖校准标签，Mondrian 分区的合法性更干净。
    """
    r = np.asarray(reference, float)
    r = r[np.isfinite(r)]
    if r.size < n_bins * 2:
        return np.array([-np.inf, np.inf])
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(r, qs))
    if edges.size < 3:
        return np.array([-np.inf, np.inf])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_bins(signal: np.ndarray, edges: np.ndarray) -> np.ndarray:
    s = np.asarray(signal, float)
    idx = np.digitize(s, edges[1:-1], right=False)
    return idx


def coverage_by_group(signal, covered, edges):
    """按给定分箱边界统计各组的覆盖率，返回逐组结果与"最差组"指标。"""
    s = np.asarray(signal, float)
    cov = np.asarray(covered).astype(bool)
    g = assign_bins(s, edges)
    rows = []
    for b in np.unique(g):
        m = g == b
        cs = coverage_stats(cov[m])
        lo, hi = np.asarray(edges)[b], np.asarray(edges)[b + 1] if b + 1 < len(edges) else np.inf
        rows.append({"bin": int(b), "bin_lo": float(lo), "bin_hi": float(hi), **cs})
    if not rows:
        return rows, {}
    covs = np.array([r["coverage"] for r in rows], float)
    summary = {
        "worst_group_coverage": float(np.nanmin(covs)),
        "best_group_coverage": float(np.nanmax(covs)),
        "coverage_spread": float(np.nanmax(covs) - np.nanmin(covs)),
        "n_groups": len(rows),
    }
    return rows, summary


def group_rank_quantile(x, group_codes) -> np.ndarray:
    """把每个样本的取值换成**组内**秩分位（0..1），NaN 原样保留。

    为什么必须按组：绝对量纲在不同数据集之间不可比。
    例如 d_wstd 在某个靶点上量程 0.2、在另一个上量程 2.0，
    直接跨数据集比较等于让量程大的数据集主导结论。换成组内秩分位后，
    "某分子在其数据集内有多异常"才是可比的。
    （分析 AD / 预警规则时必须用这个，否则结论会由量纲大的数据集决定。）

    缺失值策略：**不要 dropna**。在相似度/邻居数这类量上，NaN 往往意味着
    "连一个像样的邻居都没有"，它是**最极端的那一档**，丢掉等于系统性低估最难的一端
    （实战中丢掉 4.7% 的 NaN 会让核心数字从 61.8% 掉到 57.2%）。
    调用方应显式决定：置为极值（推荐）还是单独成组。
    """
    x = np.asarray(x, float)
    g = np.asarray(group_codes)
    out = np.full(x.shape, np.nan)
    for gc in np.unique(g):
        m = g == gc
        xv = x[m]
        ok = np.isfinite(xv)
        if ok.sum() == 0:
            continue
        if ok.sum() == 1:
            out[m] = np.where(ok, 0.5, np.nan)
            continue
        r = stats.rankdata(xv[ok], method="average")
        v = np.full(xv.shape, np.nan)
        v[ok] = (r - 1) / (ok.sum() - 1)
        out[m] = v
    return out
