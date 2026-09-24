"""cliffcp —— 活动悬崖处的共形覆盖失效研究。

模块划分：
  data        数据加载与清洗（源无关，兼容 MoleculeACE）
  features    分子指纹与 Tanimoto 精确邻域
  cliffs      活动悬崖的三套定义（数据集级 / 训练邻域暴露 / 可部署代理）
  conformal   三种共形预测（global / cluster / knn_weighted）
  stats       Wilson 区间、分层置换检验、分组覆盖
"""

__version__ = "0.1.0"
