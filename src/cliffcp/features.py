"""分子特征与相似度邻域计算。

两个职责：
1. SMILES -> ECFP4 (Morgan r=2, 2048 bit) 的稠密 uint8 矩阵，供 sklearn 使用
2. 基于 Tanimoto 的相似邻域检索（分块精确扫描，不截断 top-k 之外的真邻居）

为什么邻域检索要"精确"而不是 top-k：
活动悬崖的判据是"存在一个 sim>=T_sim 的邻居，且 |dy|>=T_act"。
若只取 top-k，在化学空间稠密区会漏掉合法邻居 —— 稠密区恰恰是悬崖高发区，
漏检会让结论偏向"无效应"。所以这里用阈值掩码扫描。
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def canonicalize(smiles: str) -> str | None:
    """返回标准 SMILES；解析失败返回 None。"""
    if smiles is None:
        return None
    s = str(smiles).strip()
    if not s:
        return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def morgan_fps(smiles_list, radius: int = 2, n_bits: int = 2048):
    """SMILES 列表 -> (fps 列表, 有效索引列表)。无效分子被跳过。"""
    fps, keep = [], []
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        fps.append(fp)
        keep.append(i)
    return fps, np.asarray(keep, dtype=np.int64)


def fps_to_matrix(fps, n_bits: int = 2048) -> np.ndarray:
    """RDKit ExplicitBitVect 列表 -> (n, n_bits) uint8 矩阵。

    优先用 DataStructs.ConvertToNumpyArray（C++ 直填，最快）；
    失败则退回位串解码路径。
    """
    n = len(fps)
    out = np.zeros((n, n_bits), dtype=np.uint8)
    try:
        for i, fp in enumerate(fps):
            DataStructs.ConvertToNumpyArray(fp, out[i])
        return out
    except Exception:
        out = np.zeros((n, n_bits), dtype=np.uint8)
        for i, fp in enumerate(fps):
            bits = fp.ToBitString()
            out[i] = np.frombuffer(bits.encode(), dtype=np.uint8) - ord("0")
        return out


def rdkit_desc_matrix(smiles_list, verbose: bool = True):
    """SMILES -> RDKit 2D 描述符矩阵（约 210 维，列名排序保证可复现）。

    这是与 ECFP 完全不同的分子表示，用来检验结论是不是指纹表示的产物。
    注意：**悬崖判定与邻域检索仍然一律用 ECFP 指纹**，
    只有"预测器看到的特征"换成描述符 —— 这样才能把"表示"这一个变量单独隔离出来。

    缺失值处理：非有限值先置 NaN，再用**全数据集**的列中位数填补。
    用的是特征统计量、不涉及标签，因此不构成标签泄漏；但属于直推式填补，
    论文里要如实说明（若审稿人要求，可改成只用训练集统计量，代价是要把填补搬进划分循环）。
    """
    from rdkit.Chem import Descriptors

    names = None
    rows = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            rows.append(None)
            continue
        d = Descriptors.CalcMolDescriptors(m, missingVal=np.nan)
        if names is None:
            names = sorted(d.keys())
        rows.append(np.array([d[k] for k in names], dtype=np.float64))

    if names is None:
        raise ValueError("no valid molecules")

    X = np.full((len(rows), len(names)), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        if r is not None:
            X[i] = r
    X[~np.isfinite(X)] = np.nan

    # 1) 缺失值：用全数据集列中位数填补（只用特征统计量，不涉及标签）
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    idx = np.where(~np.isfinite(X))
    X[idx] = np.take(med, idx[1])

    # 2) 缩尾：RDKit 里有若干量级极大的描述符（Ipc、BCUT 等），会超出 float32 表示范围，
    #    转 float32 时变成 inf，sklearn 的 RandomForest 会直接拒收
    #    （实测报错 "Input X contains infinity or a value too large for dtype('float32')"）。
    #    按列做 0.5%/99.5% 缩尾，既解决溢出，也顺手压掉极端离群值的杠杆。
    lo = np.percentile(X, 0.5, axis=0)
    hi = np.percentile(X, 99.5, axis=0)
    X = np.clip(X, lo, hi)
    X = np.clip(X, -np.finfo(np.float32).max, np.finfo(np.float32).max)

    # 3) 删掉缩尾后变成常数的列（无判别力，且会拖慢树模型）
    keep_cols = np.nanstd(X, axis=0) > 0
    X = X[:, keep_cols]

    if verbose:
        print(
            f"  [desc] {X.shape[0]} x {X.shape[1]} RDKit 2D descriptors"
            f"（原始 {len(names)} 列，缩尾后去常数 {len(names) - int(keep_cols.sum())} 列）",
            flush=True,
        )
    return X.astype(np.float32), [n for n, k in zip(names, keep_cols) if k]


def tanimoto_neighbors(
    fps,
    sim_threshold: float = 0.85,
    chunk: int = 256,
    max_neighbors: int = 512,
    verbose: bool = False,
):
    """精确找出每对 Tanimoto >= sim_threshold 的邻居（不含自身）。

    返回 (neigh_idx, neigh_sim)，两者均为长度为 n 的 object 数组，
    第 i 项是该分子的邻居索引数组与对应相似度数组（已按相似度降序）。

    实现要点：分块只是为了让每次 BulkTanimotoSimilarity 调用的输入规模可控，
    **每个查询分子必须扫过全部块**。早期版本把内外层都压在同一个块上，
    结果只找到块内邻居、跨块邻居全丢 —— 在化学空间稠密区（悬崖高发区）
    这种漏检会把现象系统性抹平，是本项目最危险的一类 bug。
    """
    n = len(fps)
    neigh_idx = np.empty(n, dtype=object)
    neigh_sim = np.empty(n, dtype=object)

    for q0 in range(0, n, chunk):
        q1 = min(q0 + chunk, n)
        acc_idx = [None] * (q1 - q0)
        acc_sim = [None] * (q1 - q0)

        for b0 in range(0, n, chunk):
            b1 = min(b0 + chunk, n)
            block = fps[b0:b1]
            for oi, i in enumerate(range(q0, q1)):
                sims = np.asarray(
                    DataStructs.BulkTanimotoSimilarity(fps[i], block),
                    dtype=np.float32,
                )
                if b0 <= i < b1:
                    sims[i - b0] = -1.0  # 排除自身
                hit = np.flatnonzero(sims >= sim_threshold)
                if hit.size == 0:
                    continue
                new_idx = (b0 + hit).astype(np.int64)
                new_sim = sims[hit]
                if acc_idx[oi] is None:
                    acc_idx[oi] = new_idx
                    acc_sim[oi] = new_sim
                else:
                    acc_idx[oi] = np.concatenate([acc_idx[oi], new_idx])
                    acc_sim[oi] = np.concatenate([acc_sim[oi], new_sim])

        for oi, i in enumerate(range(q0, q1)):
            if acc_idx[oi] is None:
                neigh_idx[i] = np.empty(0, dtype=np.int64)
                neigh_sim[i] = np.empty(0, dtype=np.float32)
            else:
                ss = acc_sim[oi]
                order = np.argsort(-ss, kind="stable")[:max_neighbors]
                neigh_idx[i] = acc_idx[oi][order]
                neigh_sim[i] = ss[order]

        if verbose:
            print(f"  neighbor scan {q1}/{n}", flush=True)

    return neigh_idx, neigh_sim


# ---------------------------------------------------------------- 骨架划分（P0）
# 随机划分下可交换性几乎平凡成立，是"cliff 处覆盖失效"最容易被审稿人攻击的一点。
# 用 Bemis-Murcko 骨架做分组划分：同一骨架的分子不跨 train/cal/test，
# 制造真实的化学空间偏移（新骨架 = 新的化学型），是最小成本的"现实性"补丁。
# 更强的做法是偏移强度谱（随机 -> scaffold -> 活性/时间偏移），见后续工作。


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko 骨架 SMILES；无环分子返回空串；解析失败返回 None。"""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return ""


def scaffold_groups(smiles_list):
    """按骨架分组，返回 [[idx, ...], ...]，组按大小降序（大骨架优先归训练集）。

    这是 scaffold split 的标准做法：把同一骨架的分子整体分配到同一折，
    避免"结构近邻横跨 train/test"造成的乐观偏差。
    """
    from collections import defaultdict

    buckets = defaultdict(list)
    for i, s in enumerate(smiles_list):
        sc = murcko_scaffold(s)
        buckets["" if sc is None else sc].append(i)
    return [sorted(v) for v in sorted(buckets.values(), key=len, reverse=True)]


def scaffold_split_indices(smiles_list, frac_train=0.6, frac_cal=0.2):
    """返回 (train_idx, cal_idx, test_idx)：按骨架分组切分，比例尽可能贴近给定 fracs。"""
    n = len(smiles_list)
    n_train = int(round(frac_train * n))
    n_cal = int(round(frac_cal * n))
    train, cal, test = [], [], []
    for grp in scaffold_groups(smiles_list):
        if len(train) + len(grp) <= n_train:
            train.extend(grp)
        elif len(cal) + len(grp) <= n_cal:
            cal.extend(grp)
        else:
            test.extend(grp)
    return (np.asarray(sorted(train), dtype=np.int64),
            np.asarray(sorted(cal), dtype=np.int64),
            np.asarray(sorted(test), dtype=np.int64))
