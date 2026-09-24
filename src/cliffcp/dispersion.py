"""无标签预警信号：用训练集邻域的标签分歧刻画"不可预测性"。

动机
----
上一轮用 `|预测值 - 最近训练邻居标签|` 作信号很弱（整体 p 仅 0.036），原因已定位：
模型（RF）的预测被相似分子平滑过，当最近邻与测试点极相似时预测值自然靠近该邻居标签，
这个差被系统性压小 —— 信号被构造性地削弱了。

本模块改用**只看训练集**的量：候选分子在训练集里的近邻标签本身有多大分歧。
模型怎么预测完全不参与，因此不会被平滑效应削弱。

信号族（全部无标签，只用到训练集的 y 与分子指纹）
------------------------------------------------
  d_nb         局部密度：sim >= sim_floor 的训练邻居个数
  d_wstd       相似度加权的近邻标签标准差      <- 主候选
  d_std        近邻标签标准差
  d_range      近邻标签极差 (max - min)
  d_iqr        近邻标签四分位距
  d_pair_gap   邻域内是否存在活动悬崖对：即是否存在两个互相相似(sim>=pair_sim)
               但标签差 >= pair_act 的训练邻居。这是最贴近机制的量 ——
               它直接说"这块化学空间的标签本身是矛盾的"。
  d_nn_sim     最近训练邻居相似度（适用于域指标，作为对照）

与 cliffs.py 中 Def C 的区别：Def C 用了模型的预测值；这里全部基于训练集本身。
"""

from __future__ import annotations

import numpy as np
from rdkit import DataStructs


def neighbor_label_stats(
    fps,
    y,
    train_idx,
    target_idx,
    neigh_idx,
    sim_floor: float = 0.5,
    k: int = 50,
    pair_sim: float = 0.85,
    pair_act: float = 1.0,
    verbose: bool = False,
):
    """为 target_idx 中每个分子计算训练集邻域标签统计量。

    neigh_idx 是预先算好的全局近邻表（阈值 pair_sim），用于快速判断
    "两个训练邻居是否互相相似"，避免在 Python 里重算两两 Tanimoto。

    返回 dict，各数组与 target_idx 等长。
    """
    y = np.asarray(y, dtype=np.float64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    target_idx = np.asarray(target_idx, dtype=np.int64)
    n_all = len(y)

    train_mask = np.zeros(n_all, dtype=bool)
    train_mask[train_idx] = True
    y_train = y[train_idx]
    train_fps = [fps[i] for i in train_idx]

    local_flags = np.zeros(n_all, dtype=bool)

    m = target_idx.size
    out = {name: np.full(m, np.nan) for name in
           ["d_nb", "d_wstd", "d_std", "d_range", "d_iqr", "d_pair_gap", "d_nn_sim"]}

    for oi, i in enumerate(target_idx):
        sims = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fps[i], train_fps), dtype=np.float64
        )
        sel = np.flatnonzero(sims >= sim_floor)
        if sel.size == 0:
            out["d_nb"][oi] = 0
            out["d_pair_gap"][oi] = 0.0
            continue

        # 取相似度最高的 k 个
        order = np.argsort(-sims[sel], kind="stable")[:k]
        sel = sel[order]
        s = sims[sel]
        yy = y_train[sel]

        out["d_nb"][oi] = float(sel.size)
        out["d_nn_sim"][oi] = float(s[0])

        # 相似度加权标准差
        wsum = s.sum()
        if wsum > 0:
            mu = float((s * yy).sum() / wsum)
            var = float((s * (yy - mu) ** 2).sum() / wsum)
            out["d_wstd"][oi] = float(np.sqrt(max(var, 0.0)))
        out["d_std"][oi] = float(yy.std()) if sel.size > 1 else 0.0
        out["d_range"][oi] = float(yy.max() - yy.min()) if sel.size > 1 else 0.0
        if sel.size >= 4:
            q1, q3 = np.percentile(yy, [25, 75])
            out["d_iqr"][oi] = float(q3 - q1)

        # 邻域内是否存在活动悬崖对：
        # 在本邻域内找两个互相相似(全局近邻表已按 pair_sim 过滤)、但标签差最大的训练分子
        gap = 0.0
        if neigh_idx is not None:
            local_flags[:] = False
            local_flags[train_idx[sel]] = True
            for a_pos in range(sel.size):
                a_global = train_idx[sel[a_pos]]
                ja = neigh_idx[a_global]
                if ja is None or len(ja) == 0:
                    continue
                hit = ja[local_flags[ja]]
                if hit.size == 0:
                    continue
                g = float(np.max(np.abs(y[hit] - y[a_global])))
                if g > gap:
                    gap = g
        out["d_pair_gap"][oi] = float(gap)

        if verbose and (oi + 1) % 500 == 0:
            print(f"    stats {oi+1}/{m}", flush=True)

    return out


def detector_matrix(stats: dict):
    """把统计量组装成 (n, n_detectors) 矩阵与名称列表，按"认为越不可靠越靠前"排序。"""
    names = ["d_wstd", "d_std", "d_range", "d_iqr", "d_pair_gap", "d_nb", "d_nn_sim"]
    cols = [stats[n] for n in names]
    M = np.column_stack(cols)
    return M, names
