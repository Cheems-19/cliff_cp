"""活动悬崖 (activity cliff, AC) 的三套定义。

为什么需要三套：
- Def A（数据集级）：与文献（MoleculeACE / SALI 文献）可比，回答"这个分子是不是悬崖分子"。
- Def B（训练邻域暴露）：与机制挂钩。CP 的失效机制是"校准集里相似的邻居给出不具代表性的标签"，
  所以真正相关的是"该测试分子在训练集里有没有一个非常相似但活性差很大的邻居"。
- Def C（可部署代理）：Def B 用了测试分子的真实标签，只能做诊断，不能上线。
  用模型自身预测与最近训练邻居标签的背离代替，得到无标签的预警信号，
  这既是"修复机制"的雏形，也是把诊断变成方法的入口。

定义统一用 (sim_threshold, act_threshold) 两个阈值：
  AC 对 := Tanimoto >= sim_threshold 且 |dy| >= act_threshold
经典取法为 sim>=0.85 且 |dpotency|>=1.0 log 单位（"强悬崖"可用 2.0）。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class CliffConfig:
    sim_threshold: float = 0.85
    act_threshold: float = 1.0
    strong_act_threshold: float = 2.0

    def tag(self) -> str:
        return f"sim{self.sim_threshold:.2f}_act{self.act_threshold:.1f}"

    def to_dict(self):
        return asdict(self)


def dataset_level_cliffs(neigh_idx, neigh_sim, y, cfg: CliffConfig):
    """Def A：分子只要参与至少一对 AC 对，就标记为悬崖分子。

    返回 dict:
      is_cliff        (n,) bool
      n_cliff_pairs   (n,) int   每个分子所处的 AC 对数
      sali_max        (n,) float 该分子所有 AC 对里最大的 SALI
      is_cliff_strong 同上但用更强的活性差阈值
    """
    n = len(y)
    is_cliff = np.zeros(n, dtype=bool)
    is_cliff_strong = np.zeros(n, dtype=bool)
    n_pairs = np.zeros(n, dtype=np.int64)
    sali_max = np.zeros(n, dtype=np.float64)

    for i in range(n):
        nb = neigh_idx[i]
        if nb.size == 0:
            continue
        ns = neigh_sim[i]
        dy = np.abs(y[nb] - y[i])

        hit = (ns >= cfg.sim_threshold) & (dy >= cfg.act_threshold)
        if not np.any(hit):
            continue
        is_cliff[i] = True
        n_pairs[i] = int(hit.sum())
        sali = dy[hit] / np.maximum(1.0 - ns[hit], 1e-6)
        sali_max[i] = float(sali.max())
        if np.any(dy[hit] >= cfg.strong_act_threshold):
            is_cliff_strong[i] = True

    return {
        "is_cliff": is_cliff,
        "is_cliff_strong": is_cliff_strong,
        "n_cliff_pairs": n_pairs,
        "sali_max": sali_max,
    }


def train_neighbor_exposure(
    test_idx,
    train_idx,
    neigh_idx,
    neigh_sim,
    y,
    pred,
    cfg: CliffConfig,
):
    """Def B / Def C：对每个测试分子，找它在训练集里的最近邻，计算暴露量。

    返回 dict（键为全局分子索引）:
      nn_sim       最近训练邻居的 Tanimoto；无训练邻居则为 nan
      nn_y_diff    |y_i - y_nn|      （需要真实标签，仅用于诊断）
      nn_pred_diff |pred_i - y_nn|   （无标签，可部署）
      sali_oracle  |y_i - y_nn| / (1 - nn_sim)
      exposed      Def B：nn_sim>=T_sim 且 nn_y_diff>=T_act
      exposed_proxy Def C：nn_sim>=T_sim 且 nn_pred_diff>=T_act
    """
    n = len(y)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True

    nn_sim = np.full(n, np.nan)
    nn_y_diff = np.full(n, np.nan)
    nn_pred_diff = np.full(n, np.nan)
    sali_oracle = np.full(n, np.nan)

    for i in test_idx:
        nb = neigh_idx[i]
        if nb.size == 0:
            continue
        in_train = train_mask[nb]
        if not np.any(in_train):
            continue
        # 邻居已按相似度降序，第一个落在训练集的就是"最近训练邻居"
        j = nb[in_train][0]
        s = float(neigh_sim[i][in_train][0])
        nn_sim[i] = s
        nn_y_diff[i] = abs(float(y[i]) - float(y[j]))
        nn_pred_diff[i] = abs(float(pred[i]) - float(y[j]))
        sali_oracle[i] = nn_y_diff[i] / max(1.0 - s, 1e-6)

    exposed = (nn_sim >= cfg.sim_threshold) & (nn_y_diff >= cfg.act_threshold)
    exposed_proxy = (nn_sim >= cfg.sim_threshold) & (nn_pred_diff >= cfg.act_threshold)

    return {
        "nn_sim": nn_sim,
        "nn_y_diff": nn_y_diff,
        "nn_pred_diff": nn_pred_diff,
        "sali_oracle": sali_oracle,
        "exposed": np.nan_to_num(exposed, nan=False).astype(bool),
        "exposed_proxy": np.nan_to_num(exposed_proxy, nan=False).astype(bool),
    }
