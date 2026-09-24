"""共形预测的三种实现。

三种方法的差别只在"用哪个分位数"：

- global       : 全部校准样本共用一个分位数（基线，文献里的 split CP）
- cluster      : 先聚类，每个簇用各自的分位数；簇内校准样本不足则收缩回退全局分位数（Mondrian-by-cluster）
- knn_weighted : 每个测试点用"与各校准点的相似度"当权重，取加权分位数（localized CP, Hore & Barber 风格）

预设的结论（待验证）：cluster / knn_weighted 会改善非悬崖组的覆盖，
但对悬崖组无效甚至更差 —— 因为悬崖邻域内标签本身矛盾，局部化把更不可靠的邻居权重提上去了。
如果实测成立，这就是本文最有引用价值的一条。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 分位数


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """标准 split CP 分位数：第 ceil((n+1)(1-alpha)) 小的校准分数。"""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = s.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


def weighted_quantile(scores: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """加权分位数：排序后找累计归一化权重首次达到 1-alpha 的位置。"""
    s = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    keep = w > 0
    s, w = s[keep], w[keep]
    if s.size == 0:
        return float("inf")
    order = np.argsort(s, kind="stable")
    s, w = s[order], w[order]
    cum = np.cumsum(w) / w.sum()
    idx = int(np.searchsorted(cum, 1.0 - alpha))
    return float(s[min(idx, s.size - 1)])


# ---------------------------------------------------------------- 方法


def global_intervals(pred_cal, y_cal, pred_test, alpha):
    """split CP（绝对残差）。"""
    scores = np.abs(np.asarray(y_cal, float) - np.asarray(pred_cal, float))
    q = conformal_quantile(scores, alpha)
    p = np.asarray(pred_test, float)
    return {
        "q": np.full(p.shape, q),
        "lower": p - q,
        "upper": p + q,
    }


def cqr_intervals(
    X_train,
    y_train,
    X_cal,
    y_cal,
    X_test,
    alpha: float = 0.1,
    seed: int = 0,
    max_iter: int = 400,
):
    """共形分位数回归（CQR, Romano et al. 2019）——最强的常规对手。

    两步：
      1) 在训练集上拟合两个分位数回归器，分别预测 alpha/2 与 1-alpha/2 条件分位数；
      2) 用校准集上的 CQR 不一致分数 E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
         取 (1-alpha) 经验分位数 q，最终区间 [q_lo - q, q_hi + q]。

    为什么必须放进对比：CQR 的区间**不对称**且随 x 变化，
    它本身就在做"按难度自适应宽度"。如果本文的分区方法只赢了绝对残差基线，
    审稿人有理由说"你没跟真正的强基线比"。这里正面比。

    注意：CQR 不是"由单个 q 缩放"的规则，因此不参与 rule_explore / rule_frontier
    的离线模拟；它只作为 METHODS.csv 里的一行真实对手出现。
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    Xtr = np.asarray(X_train, np.float32)
    Xc = np.asarray(X_cal, np.float32)
    Xt = np.asarray(X_test, np.float32)
    ytr = np.asarray(y_train, float)
    yc = np.asarray(y_cal, float)

    lo = HistGradientBoostingRegressor(
        loss="quantile", quantile=alpha / 2.0,
        random_state=seed, max_iter=max_iter,
    )
    hi = HistGradientBoostingRegressor(
        loss="quantile", quantile=1.0 - alpha / 2.0,
        random_state=seed, max_iter=max_iter,
    )
    lo.fit(Xtr, ytr)
    hi.fit(Xtr, ytr)

    e = np.maximum(lo.predict(Xc) - yc, yc - hi.predict(Xc))
    q = conformal_quantile(e, alpha)

    lo_t, hi_t = lo.predict(Xt), hi.predict(Xt)
    lower = lo_t - q
    upper = hi_t + q
    return {
        "q": 0.5 * (upper - lower),
        "lower": lower,
        "upper": upper,
    }


def cluster_intervals(
    X_fit,
    pred_cal,
    y_cal,
    X_test,
    pred_test,
    alpha,
    n_clusters: int = 12,
    min_cal_per_cluster: int = 30,
    seed: int = 0,
):
    """Mondrian-by-cluster：簇条件化校准 + 样本不足时回退全局。

    X_fit   : 用于聚类的特征矩阵（建议用训练集的指纹矩阵）
    X_test  : 测试点特征矩阵（同一空间）
    """
    from sklearn.cluster import MiniBatchKMeans

    X_fit = np.asarray(X_fit)
    X_test = np.asarray(X_test)

    n_clusters = int(max(2, min(n_clusters, max(2, len(pred_cal) // max(1, min_cal_per_cluster)))))
    km = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed, n_init=3, batch_size=1024
    )
    km.fit(X_fit)

    cal_lab = km.predict(X_fit)
    test_lab = km.predict(X_test)

    scores = np.abs(np.asarray(y_cal, float) - np.asarray(pred_cal, float))
    q_global = conformal_quantile(scores, alpha)

    q_map = {}
    used_global = 0
    for c in range(n_clusters):
        m = cal_lab == c
        if int(m.sum()) >= min_cal_per_cluster:
            q_map[c] = conformal_quantile(scores[m], alpha)
        else:
            q_map[c] = q_global
            used_global += 1

    q_test = np.array([q_map[int(c)] for c in test_lab], dtype=np.float64)
    p = np.asarray(pred_test, float)

    return {
        "q": q_test,
        "lower": p - q_test,
        "upper": p + q_test,
        "n_clusters": n_clusters,
        "n_clusters_fallback": used_global,
        "q_global": q_global,
        "test_cluster": test_lab,
    }


def knn_weighted_intervals(
    sim_test_cal,
    pred_cal,
    y_cal,
    pred_test,
    alpha,
    k: int = 200,
    tau: float = 0.0,
):
    """localized CP：以测试点为中心、用相似度做核权重取加权分位数。

    sim_test_cal : (n_test, n_cal) 的 Tanimoto 相似度矩阵
    """
    sim = np.asarray(sim_test_cal, dtype=np.float64)
    scores = np.abs(np.asarray(y_cal, float) - np.asarray(pred_cal, float))
    p = np.asarray(pred_test, float)

    n_test, n_cal = sim.shape
    k = int(min(k, n_cal))
    q = np.empty(n_test, dtype=np.float64)

    for i in range(n_test):
        row = sim[i]
        top = np.argpartition(-row, k - 1)[:k] if k < n_cal else np.arange(n_cal)
        w = np.clip(row[top] - tau, 0.0, None)
        if not np.any(w > 0):
            q[i] = conformal_quantile(scores, alpha)
        else:
            q[i] = weighted_quantile(scores[top], w, alpha)

    return {"q": q, "lower": p - q, "upper": p + q}


# ---------------------------------------------------------------- 评估


def empirical_coverage(lower, upper, y_test):
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    y = np.asarray(y_test, float)
    return float(np.mean((y >= lower) & (y <= upper)))


def interval_width(lower, upper, mask=None):
    w = np.asarray(upper, float) - np.asarray(lower, float)
    if not np.isfinite(w).all():
        w = np.where(np.isfinite(w), w, np.nan)
    w = w[mask] if mask is not None else w
    return float(np.nanmean(w)) if len(w) else float("nan")


# ---------------------------------------------------------------- 新增基线（P0）
# 这两条是应对审稿人"你只用 Mondrian 做条件化、选错了方法"的核心补丁：
#   - adaptive（归一化/自适应共形，Papadopoulos 2002 / MAPIE ResidualNormalisedScore）
#   - weighted（协变量偏移加权共形，Tibshirani et al. 2019）
# 均为纯加法，不改动上面既有函数，避免影响正在跑的流水线。


def adaptive_intervals(
    X_fit,
    resid_fit,
    X_cal,
    pred_cal,
    y_cal,
    X_test,
    pred_test,
    alpha: float,
    eps: float = 1e-3,
    seed: int = 0,
    max_iter: int = 200,
):
    """归一化/自适应共形：以 x 依赖的尺度 sigma(x) 归一化非一致性分数。

    score_i = |y_i - pred_i| / sigma(x_i)，区间 = pred(x) +- q * sigma(x)。

    sigma(x) 由一个辅助回归器拟合；调用方须用**留出残差**（例如 RF 的 OOB 预测）
    来拟合它，否则在训练残差上拟合会把 sigma 低估、区间过窄。
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    rfit = np.abs(np.asarray(resid_fit, float))
    scale = HistGradientBoostingRegressor(random_state=seed, max_iter=max_iter)
    scale.fit(np.asarray(X_fit, np.float32), rfit)

    def _sig(X):
        return np.maximum(eps, scale.predict(np.asarray(X, np.float32)))

    sig_cal = _sig(X_cal)
    sig_test = _sig(X_test)

    scores = np.abs(np.asarray(y_cal, float) - np.asarray(pred_cal, float)) / sig_cal
    q = conformal_quantile(scores, alpha)

    p = np.asarray(pred_test, float)
    half = q * sig_test
    return {"q": half, "q_scalar": q, "sigma_test": sig_test, "lower": p - half, "upper": p + half}


def oob_abs_residual(model, X_train, y_train):
    """RF 的 OOB 留出残差绝对值；无 oob 时退回 5 折交叉拟合。"""
    Xtr = np.asarray(X_train, np.float32)
    ytr = np.asarray(y_train, float)
    oob = getattr(model, "oob_prediction_", None)
    if oob is not None and np.isfinite(oob).any():
        r = np.abs(ytr - np.asarray(oob, float))
        if np.isfinite(r).all():
            return r
    from sklearn.model_selection import KFold
    from sklearn.base import clone

    r = np.full(len(ytr), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Xtr):
        m = clone(model)
        m.fit(Xtr[tr], ytr[tr])
        r[te] = np.abs(ytr[te] - m.predict(Xtr[te]))
    r[~np.isfinite(r)] = np.nanmedian(r)
    return r


def weighted_shift_intervals(
    X_cal,
    pred_cal,
    y_cal,
    X_test,
    pred_test,
    alpha: float,
    seed: int = 0,
    clip: float = 20.0,
    clf_kind: str = "gbm",
):
    """协变量偏移下的加权共形（Tibshirani et al. 2019）。

    用分类器估计密度比 w(x) = p_test(x)/p_cal(x)（训练"校准 vs 测试"判别器，
    取似然比），对校准分数做**加权**分位数。随机划分下 w ≈ 1（退化为 global）；
    scaffold 划分下 w 承担真实偏移，是"换划分"下最该有的对手。

    clf_kind: "gbm"（默认，能捕捉非线性偏移）| "logit"（线性，更快更稳）。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    Xc = np.asarray(X_cal, np.float32)
    Xt = np.asarray(X_test, np.float32)
    Z = np.vstack([Xc, Xt])
    lab = np.concatenate([np.zeros(len(Xc)), np.ones(len(Xt))])

    sc = StandardScaler()
    Z = sc.fit_transform(Z)
    if clf_kind == "logit":
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, max_depth=3, random_state=seed
        )
    clf.fit(Z, lab)

    p_cal = clf.predict_proba(Z[: len(Xc)])[:, 1]
    w = clip * p_cal / np.clip(1.0 - p_cal, 1e-6, 1.0)

    scores = np.abs(np.asarray(y_cal, float) - np.asarray(pred_cal, float))
    q = weighted_quantile(scores, w, alpha)

    p = np.asarray(pred_test, float)
    return {"q": np.full(p.shape, q), "q_scalar": q, "lower": p - q, "upper": p + q}

