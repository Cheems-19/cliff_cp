"""数据加载与清洗。

设计原则：源无关。任何含 (smiles, 连续活性值) 的 CSV 都能吃，
列名自动识别，兼容 MoleculeACE 的 benchmark_data 格式。
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from .features import canonicalize

SMILES_ALIASES = [
    "smiles", "SMILES", "canonical_smiles", "smiles_canonical",
    "smiles_r", "smiles_c", "Smiles", "structure",
]

# 顺序即优先级。MoleculeACE 的 CSV 同时有 y (=-log10(nM)) 和 y [pEC50/pKi]，
# 必须优先取后者：前者越小越强，用它做效力分箱方向会反；
# 虽然 |dy| 在两列上相同（pEC50 = 9 + y），但可读性与分箱正确性要求用 pEC50。
TARGET_ALIASES = [
    "y [pEC50/pKi]", "pEC50/pKi", "y [pIC50/pKi]", "y [pChEMBL]",
    "pIC50", "pic50", "pEC50", "pKi", "pKd", "pChEMBL",
    "activity", "Activity", "y", "Y", "target", "value", "logS",
]
CLIFF_ALIASES = ["cliff_mol", "cliff", "Cliff", "is_cliff", "activity_cliff"]


def _pick(cols, aliases):
    for a in aliases:
        if a in cols:
            return a
    lower = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def load_potency_csv(path: str, verbose: bool = True) -> pd.DataFrame:
    """读入 CSV，返回列: smiles_canonical, smiles_raw, y[, cliff_source]。

    清洗规则：
    - SMILES 无法解析 -> 丢弃
    - 同一标准 SMILES 出现多次 -> 取活性中位数（同一化合物的多次测定）
    """
    df = pd.read_csv(path)
    cols = list(df.columns)

    s_col = _pick(cols, SMILES_ALIASES)
    y_col = _pick(cols, TARGET_ALIASES)
    if s_col is None or y_col is None:
        raise ValueError(
            f"无法识别列。smiles_col={s_col} target_col={y_col} 实际列={cols}"
        )
    c_col = _pick(cols, CLIFF_ALIASES)

    raw = df[[s_col, y_col]].copy()
    raw.columns = ["smiles_raw", "y"]
    raw["y"] = pd.to_numeric(raw["y"], errors="coerce")
    raw = raw.dropna(subset=["y"])
    if c_col is not None:
        raw["cliff_source"] = df.loc[raw.index, c_col].values

    raw["smiles_canonical"] = [canonicalize(s) for s in raw["smiles_raw"]]
    n0 = len(raw)
    raw = raw.dropna(subset=["smiles_canonical"])

    # 同一化合物多次测定 -> 中位数
    agg = {"y": "median", "smiles_raw": "first"}
    if "cliff_source" in raw.columns:
        agg["cliff_source"] = "max"
    out = (
        raw.groupby("smiles_canonical", as_index=False)
        .agg(agg)
        .reset_index(drop=True)
    )

    if verbose:
        print(
            f"[data] {os.path.basename(path)}: 原始 {n0} 行 -> 有效 {len(out)} 个唯一化合物"
            f" | y 范围 [{out['y'].min():.2f}, {out['y'].max():.2f}]",
            flush=True,
        )
    return out


def find_moleculeace_datasets(root: str):
    """在 MoleculeACE 目录里找出所有 benchmark 数据文件。

    优先只扫 `benchmark_data` 子目录：仓库里还有 results/ 等目录，
    里面的 CSV 可能恰好也有 smiles+target 列（例如模型输出），
    一并扫进来会污染统计。只有找不到 benchmark_data 时才全树扫描。
    """
    preferred = None
    for dirpath, dirnames, _filenames in os.walk(root):
        if os.path.basename(dirpath) == "benchmark_data":
            preferred = dirpath
            break

    search_dirs = [preferred] if preferred else [root]

    hits = []
    for base in search_dirs:
        if not preferred:
            iterator = (
                os.path.join(dp, fn)
                for dp, _dn, fns in os.walk(base)
                for fn in fns
            )
        else:
            iterator = (
                os.path.join(base, fn)
                for fn in os.listdir(base)
            )
        for p in iterator:
            if not p.lower().endswith(".csv"):
                continue
            try:
                head = pd.read_csv(p, nrows=1)
            except Exception:
                continue
            cols = list(head.columns)
            if _pick(cols, SMILES_ALIASES) and _pick(cols, TARGET_ALIASES):
                hits.append(p)
    return sorted(hits)
