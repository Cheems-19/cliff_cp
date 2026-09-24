"""Chemprop v2（D-MPNN）作为第三个模型族的接入层。

设计要点（为什么这样做而不是别的）：

1. **只用 CLI，不碰 Python API**。chemprop v2 的 Python API 在 2.3.x 里变动频繁，
   而 CLI 稳定且自带 `--splits-column`（可以喂进我们自己切好的划分）。
   用 subprocess 调 CLI，出错信息也更完整。

2. **模型侧不确定性用"多模型集成方差"**，与树模型的树间方差、GBR 的多种子集成方差
   保持概念一致。做法是训练 M 个模型后**逐个预测**、自己算 mean/std ——
   不依赖 chemprop 的 `--uncertainty-method` 输出列名（那个格式随版本变）。

3. **悬崖判定与邻域检索仍然用 ECFP**，只有"预测器看到的输入"换成分子图。
   这与跨架构验证的设计一致：让"模型族"成为唯一变量。

4. **不要给 val 喂测试集**。我们把自己的校准集当作 chemprop 的 val
   （它只用于 early stopping，不参与预测），测试集行标成 test，
   chemprop 会跳过它们训练但在训练结束时顺手给出 test 预测（我们不用它）。

CPU/GPU 实测参考（CHEMBL233_Ki，3142 分子，默认架构 d_h=300/depth=3）：
  CPU（机器满载）：34–127 s/epoch → 30 epoch 约 15–65 min/模型
  GPU（RTX 3090）：约 1–3 s/epoch   → 30 epoch 约 1–2 min/模型
所以 **GNN 轴必须在 GPU 上跑**；CPU 只够做单数据集的冒烟验证。
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# chemprop v2 环境目录：默认项目根下的 .conda-chemprop，可用环境变量 CHEMPROP_ENV 覆盖
_CHEMPROP_ENV = Path(os.environ.get("CHEMPROP_ENV", ".conda-chemprop"))


def set_chemprop_env(path: str | Path) -> None:
    """指定 chemprop 可执行文件所在的环境根目录（默认 .conda-chemprop）。"""
    global _CHEMPROP_ENV
    _CHEMPROP_ENV = Path(path)


def _chemprop_bin() -> Path:
    p = _CHEMPROP_ENV / "bin" / "chemprop"
    if not p.exists():
        raise FileNotFoundError(f"找不到 chemprop 可执行文件: {p}")
    return p


def _run(cmd: list[str], timeout: int = 7200) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        tail = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()[-12:]
        raise RuntimeError(
            "chemprop 调用失败（exit=%d）:\n%s" % (p.returncode, "\n".join(tail))
        )
    return p


def _read_preds(path: Path) -> pd.DataFrame:
    """读 chemprop 的预测输出，容错各种列名。"""
    df = pd.read_csv(path)
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    smiles_col = next((low[k] for k in ("smiles", "smile") if k in low), cols[0])
    pred_col = next((c for c in cols if c.lower().startswith("pred")), None)
    if pred_col is None:
        pred_col = [c for c in cols if c != smiles_col][0]
    return df[[smiles_col, pred_col]].rename(
        columns={smiles_col: "smiles", pred_col: "pred"}
    )


def train_ensemble_and_predict(
    smiles: list[str],
    y: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    *,
    n_members: int = 3,
    epochs: int = 30,
    patience: int = 5,
    seed: int = 0,
    accelerator: str = "gpu",
    gpu: str = "0",
    workdir: str | Path | None = None,
    frac_val_of_train: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """训练 M 个 D-MPNN，返回 (预测值, 模型间标准差)，按 target_idx 顺序对齐。

    参数
    ----
    smiles      : 全体分子的 SMILES（长度 n）
    y           : 全体标签（长度 n）
    train_idx   : 训练集索引（chemprop 的训练 + 内部 val 都从这里出）
    target_idx  : 需要预测的索引（通常是 cal+test 的并集）

    返回
    ----
    (pred, std) : 两个长度为 len(target_idx) 的数组。
                  std 是 M 个模型预测的样本标准差（M=1 时全 0）。

    说明
    ----
    - chemprop 自己会在 train 内再切一份做 early stopping；这里按
      `frac_val_of_train` 从 train_idx 里划出一部分当 val，
      剩下的才是真正的训练行。**cal/test 绝不进入训练或 val**。
    - 训练时用 `--splits-column` 喂划分标签，保证顺序与我们的索引一致。
    """
    import tempfile

    n = len(smiles)
    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx)
    target_idx = np.asarray(target_idx)

    # 从 train_idx 内部再划一份当 chemprop 的 val（与我们的 cal/test 完全隔离）
    perm = rng.permutation(train_idx)
    n_val = max(int(round(frac_val_of_train * len(perm))), 1)
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]

    split = np.array(["ignore"] * n, dtype=object)
    split[fit_idx] = "train"
    split[val_idx] = "val"

    wd = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="cp_"))
    wd.mkdir(parents=True, exist_ok=True)
    csv_all = wd / "all.csv"
    pd.DataFrame({"smiles": list(smiles), "y": np.asarray(y, float), "split": split}).to_csv(
        csv_all, index=False
    )

    out_dir = wd / "train"
    shutil.rmtree(out_dir, ignore_errors=True)
    cp = str(_chemprop_bin())
    _run(
        [
            cp, "train",
            "-i", str(csv_all), "-o", str(out_dir),
            "-t", "regression", "--target-columns", "y", "-s", "smiles",
            "--splits-column", "split",
            "--epochs", str(epochs), "--warmup-epochs", "1",
            "--patience", str(patience), "--ensemble-size", str(n_members),
            "--accelerator", accelerator,
            "--num-workers", "2",
            "--pytorch-seed", str(seed), "--data-seed", str(seed),
            "--remove-checkpoints",
        ]
    )

    # 逐个模型预测（M 很小，每次几秒；自己算 mean/std，不依赖其不确定性列名）
    cks = sorted(out_dir.glob("model_*/best.pt"))
    if len(cks) != n_members:
        raise RuntimeError(
            f"期望 {n_members} 个模型，只找到 {len(cks)} 个：{[c.name for c in cks]}"
        )
    preds = np.zeros((n_members, n), float)
    csv_q = wd / "query.csv"
    pd.DataFrame({"smiles": list(smiles)}).to_csv(csv_q, index=False)
    for i, ck in enumerate(cks):
        pf = wd / f"preds_{i}.csv"
        env_cmd = [cp, "predict", "-i", str(csv_q), "--model-paths", str(ck),
                   "-o", str(pf), "--accelerator", accelerator]
        _run(env_cmd)
        pr = _read_preds(pf)
        # chemprop 保持输入顺序，行数应等于 n
        if len(pr) != n:
            raise RuntimeError(f"预测行数 {len(pr)} != 输入行数 {n}")
        preds[i] = pr["pred"].to_numpy(float)

    mean = preds.mean(axis=0)
    std = preds.std(axis=0, ddof=1) if n_members > 1 else np.zeros(n)
    return mean[target_idx], std[target_idx]
