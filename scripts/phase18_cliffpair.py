# -*- coding: utf-8 -*-
"""C4：悬崖对实例图素材（自包含版）。

内部跑一个与主实验同构的最小流水线（RF200、random 60/20/20、split CP 全局区间），
在测试集中找「高相似训练邻居 + 大幅 potency 差 + 全局区间失效」的悬崖分子，
连同其邻居一起输出结构图与区间信息，供本地组合图 14。

用法（服务器）：
  .venv/bin/python scripts/phase18_cliffpair.py --dataset CHEMBL233_Ki \
      --data-dir data/MoleculeACE --out results/phase18_cliffpair --sim 0.85 --act 2.0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cliffcp import features as F  # noqa: E402
from cliffcp.conformal import conformal_quantile  # noqa: E402
from cliffcp.data import find_moleculeace_datasets, load_potency_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CHEMBL233_Ki")
    ap.add_argument("--data-dir", default="data/MoleculeACE")
    ap.add_argument("--out", default="results/phase18_cliffpair")
    ap.add_argument("--sim", type=float, default=0.85)
    ap.add_argument("--act", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = find_moleculeace_datasets(args.data_dir)
    path = next(p for p in files if args.dataset in os.path.basename(p))
    df = load_potency_csv(path)
    smiles = df.smiles_canonical.tolist()
    y = df.y.to_numpy()
    n = len(y)
    print(f"[data] {args.dataset}: {n} 分子", flush=True)

    fps, keep = F.morgan_fps(smiles)
    keep = np.asarray(keep, int)
    if not np.array_equal(keep, np.arange(n)):
        raise SystemExit("morgan_fps 跳过了无效分子（不应发生：已 canonicalize）")
    neigh_idx, neigh_sim = F.tanimoto_neighbors(fps, sim_threshold=args.sim)
    print("[neighbors] done", flush=True)

    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_tr = int(round(0.6 * n))
    n_ca = int(round(0.2 * n))
    train_idx, cal_idx, test_idx = idx[:n_tr], idx[n_tr:n_tr + n_ca], idx[n_tr + n_ca:]

    model = RandomForestRegressor(n_estimators=200, random_state=args.seed, n_jobs=8)
    model.fit(F.fps_to_matrix(fps)[train_idx], y[train_idx])
    pred = model.predict(F.fps_to_matrix(fps)[test_idx])
    scores = np.abs(y[cal_idx] - model.predict(F.fps_to_matrix(fps)[cal_idx]))
    q = conformal_quantile(scores, args.alpha)
    covered = np.abs(y[test_idx] - pred) <= q
    print(f"[cp] q_global={q:.3f}  cov_all={covered.mean():.3f}", flush=True)

    # 悬崖分子：测试分子存在训练邻居 sim≥阈值 且 |Δy|≥act
    pos = {m: k for k, m in enumerate(train_idx)}
    cands = []
    for k, ti in enumerate(test_idx):
        if covered[k]:
            continue
        nb = [(m, s) for m, s in zip(neigh_idx[ti], neigh_sim[ti]) if m in pos]
        if not nb:
            continue
        for m, s in nb:
            if abs(y[ti] - y[m]) >= args.act and y[ti] > y[m]:
                cands.append(dict(mol_i=int(ti), mol_j=int(m), sim=float(s),
                                  y_i=float(y[ti]), y_j=float(y[m]), dy=float(y[ti] - y[m]),
                                  pred_i=float(pred[k]), q_global=float(q),
                                  covered_i=False, covered_j=bool(abs(y[m] - pred[k]) <= q),
                                  err_i=float(abs(y[ti] - pred[k])),
                                  smiles_i=smiles[ti], smiles_j=smiles[m]))
    if not cands:
        raise SystemExit("无候选对（降低 --act 或换 --seed）")
    C = pd.DataFrame(cands).sort_values(["dy", "err_i"], ascending=False).drop_duplicates("mol_i")
    C.to_csv(os.path.join(args.out, "candidates.csv"), index=False)
    top = C.iloc[0]

    from rdkit import Chem
    from rdkit.Chem import Draw
    for tag, smi in [("molA_highCliff", top.smiles_i), ("molB_trainNeighbour", top.smiles_j)]:
        m = Chem.MolFromSmiles(smi)
        Draw.MolToFile(m, os.path.join(args.out, tag + ".png"), size=(460, 460))
    top.to_frame().T.to_csv(os.path.join(args.out, "pair_info.csv"), index=False)
    print("[selected]", {k: top[k] for k in ["mol_i", "mol_j", "sim", "y_i", "y_j",
                                             "pred_i", "q_global", "dy"]})
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
