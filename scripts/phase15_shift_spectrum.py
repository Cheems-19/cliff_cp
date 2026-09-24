#!/usr/bin/env python
"""Phase 15 —— C2：偏移强度谱（把「可交换性被打破」从断言变成测量）。

四个划分协议，按结构/标签偏移强度**递增**排列：

  P0  random        基准：构造上近似可交换。
  P1  scaffold      Bemis–Murcko **精确**骨架分组留出（同骨架不跨折）——温和的结构偏移。
  P2  scaffold_gen  Bemis–Murcko **泛化**骨架（原子类型→C、键级→单键）分组留出
                    —— 分组更粗、留出组更杂，偏移更强。
  P3  act_shift     活性偏移：测试集 = 效力最高的 20%（标签与协变量双偏移，
                    模拟"优化到更高效力区间"的实际外推场景）。

另外，对每个**测试分子**记录 `nn_sim_train`（到训练集的最大 Tanimoto 相似度），
供事后按「距离强度」分层、画出连续的覆盖失效曲线。

用法：
  python scripts/phase15_shift_spectrum.py --data-dir data/MoleculeACE \
      --out results/phase15_spectrum --seeds 3 --protocols random,scaffold,scaffold_gen,act_shift \
      --datasets CHEMBL204_Ki,...
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rdkit import Chem, DataStructs  # noqa: E402

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold as _MS  # noqa: E402
except Exception:                                          # pragma: no cover
    _MS = None
_GEN = getattr(_MS, "MakeScaffoldGeneric", None) if _MS is not None else None

from cliffcp import cliffs as CL  # noqa: E402
from cliffcp import conformal as CP  # noqa: E402
from cliffcp import dispersion as DP  # noqa: E402
from cliffcp import features as F  # noqa: E402
from cliffcp.data import find_moleculeace_datasets, load_potency_csv  # noqa: E402
import phase1_signal as P1  # noqa: E402
import phase11_shift as P11  # noqa: E402  (复用 bins_by_cal / rank_auc / 精确骨架划分)

PROTOCOLS = ["random", "scaffold", "scaffold_gen", "act_mid20", "act_shift40", "act_shift20"]
METHODS = ["global", "adaptive", "mondrian_wstd", "mondrian_rfstd", "mondrian_nb"]


# ---------------------------------------------------------------- 划分构造
def generic_scaffold_groups(smiles_list):
    """泛化 Bemis–Murcko 骨架分组（原子类型→C、键级→单键）。"""
    if _GEN is None:
        return None
    buckets: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        key = ""
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            try:
                key = Chem.MolToSmiles(_GEN(_MS.GetScaffoldForMol(m)))
            except Exception:
                key = ""
        buckets.setdefault(key, []).append(i)
    return [np.asarray(v, np.int64) for v in buckets.values()]


def group_split_seeded(groups, n, seed, frac_train=0.6, frac_cal=0.2):
    """给定分组，打乱组顺序后贪心分配（组内不拆）。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n_train = int(round(frac_train * n))
    n_cal = int(round(frac_cal * n))
    train, cal, test = [], [], []
    for gi in order:
        grp = groups[gi]
        if len(train) + len(grp) <= n_train:
            train.extend(grp)
        elif len(cal) + len(grp) <= n_cal:
            cal.extend(grp)
        else:
            test.extend(grp)
    return (np.asarray(sorted(train), np.int64),
            np.asarray(sorted(cal), np.int64),
            np.asarray(sorted(test), np.int64))


def act_mid_split(y, seed, frac_test=0.2, frac_train=0.6, frac_cal=0.2):
    """活性插值划分：测试集 = 效力**中位带**（50%±frac_test/2 分位区间）；
    train/cal 从其余分子按比例切。这是 act_shift 的插值对照——预期名义保证不塌。"""
    n = len(y)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(n), np.asarray(y, float)))   # y 升序
    lo = int(round((0.5 - frac_test / 2) * n))
    hi = int(round((0.5 + frac_test / 2) * n))
    test = np.sort(order[lo:hi])
    rest = rng.permutation(np.concatenate([order[:lo], order[hi:]]))
    n_rest = len(rest)
    n_train = int(round(frac_train / (frac_train + frac_cal) * n_rest))
    train = np.sort(rest[:n_train])
    cal = np.sort(rest[n_train:])
    return (np.asarray(sorted(train), np.int64),
            np.asarray(sorted(cal), np.int64),
            np.asarray(sorted(test), np.int64))


def act_shift_split(y, seed, frac_test=0.2, frac_train=0.6, frac_cal=0.2):
    """活性偏移划分：测试集 = 效力最高的 frac_test 比例；
    剩余分子按 frac_train : frac_cal 的**比例**切成 train/cal（保证 cal 不为空）。"""
    n = len(y)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(n), -np.asarray(y, float)))   # y 降序，同值随机打破
    n_test = int(round(frac_test * n))
    test = np.sort(order[:n_test])
    rest = rng.permutation(order[n_test:])
    ratio = frac_train / (frac_train + frac_cal)
    n_train = int(round(ratio * len(rest)))
    train = np.sort(rest[:n_train])
    cal = np.sort(rest[n_train:])
    return train, cal, test


def max_sim_to_train(fps, train_idx, query_idx):
    """每个查询分子到训练集的最大 Tanimoto 相似度。"""
    ref = [fps[i] for i in train_idx]
    out = np.empty(len(query_idx), np.float64)
    for j, qi in enumerate(query_idx):
        sims = DataStructs.BulkTanimotoSimilarity(fps[qi], ref)
        out[j] = float(np.max(sims)) if sims else np.nan
    return out


# ---------------------------------------------------------------- 主循环
def run_one_dataset(df, name, cfg, args, out_root):
    fps, keep = F.morgan_fps(df["smiles_canonical"].tolist())
    if len(fps) < args.min_size:
        print(f"  [skip] N={len(fps)} < {args.min_size}", flush=True)
        return None, None
    df = df.iloc[keep].reset_index(drop=True)
    y = df["y"].to_numpy(np.float64)
    smiles = df["smiles_canonical"].tolist()
    X = F.fps_to_matrix(fps, args.n_bits)
    n = len(fps)

    neigh_idx, neigh_sim = F.tanimoto_neighbors(
        fps, sim_threshold=cfg.sim_threshold, max_neighbors=args.max_neighbors
    )
    cl = CL.dataset_level_cliffs(neigh_idx, neigh_sim, y, cfg)
    n_cliff = int(cl["is_cliff"].sum())
    print(f"  [data] n={n} cliffA={n_cliff} ({n_cliff / n:.1%})", flush=True)
    if n_cliff < args.min_cliff:
        print(f"  [skip] cliff too few ({n_cliff})", flush=True)
        return None, None

    groups_exact = F.scaffold_groups(smiles)
    args._gen_warned = getattr(args, "_gen_warned", False)
    groups_gen = generic_scaffold_groups(smiles) if "scaffold_gen" in args.protocols else None
    if "scaffold_gen" in args.protocols and groups_gen is None:
        print("  [warn] generic scaffold 不可用（见上方 import 警告），本次将跳过 scaffold_gen", flush=True)
    if groups_gen is not None:
        sizes = np.array([len(g) for g in groups_gen])
        print(f"  [groups] exact: {len(groups_exact)} 组(中位 {np.median([len(g) for g in groups_exact]):.0f}) | "
              f"generic: {len(groups_gen)} 组(中位 {np.median(sizes):.0f}, 最大 {sizes.max()})", flush=True)

    m_rows, molf_rows = [], []
    rng_master = np.random.default_rng(args.seed0)

    for proto in args.protocols:
        for s in range(args.seeds):
            seed = int(rng_master.integers(1, 10 ** 8))
            if proto == "random":
                perm = np.random.default_rng(seed).permutation(n)
                nt = int(round(args.frac_train * n)); nc = int(round(args.frac_cal * n))
                train_idx, cal_idx, test_idx = perm[:nt], perm[nt:nt + nc], perm[nt + nc:]
            elif proto == "scaffold":
                train_idx, cal_idx, test_idx = P11.scaffold_split_seeded(
                    smiles, seed, args.frac_train, args.frac_cal)
            elif proto == "scaffold_gen":
                if groups_gen is None:
                    if not args._gen_warned:
                        print("  [warn] rdkit 缺少 MakeScaffoldGeneric，scaffold_gen 协议不可用", flush=True)
                        args._gen_warned = True
                    continue
                train_idx, cal_idx, test_idx = group_split_seeded(
                    groups_gen, n, seed, args.frac_train, args.frac_cal)
            elif proto.startswith("act_shift"):
                ft = float(proto.replace("act_shift", "")) / 100.0
                train_idx, cal_idx, test_idx = act_shift_split(
                    y, seed, ft, args.frac_train, args.frac_cal)
            elif proto.startswith("act_mid"):
                ft = float(proto.replace("act_mid", "")) / 100.0
                train_idx, cal_idx, test_idx = act_mid_split(
                    y, seed, ft, args.frac_train, args.frac_cal)
            else:
                continue

            if len(test_idx) < 50 or len(cal_idx) < 50:
                print(f"    [skip] {proto} seed={seed} n_test={len(test_idx)} n_cal={len(cal_idx)}", flush=True)
                continue

            model = P1.make_model(args.model, seed)
            model.fit(X[train_idx].astype(np.float32), y[train_idx])
            pred = model.predict(X.astype(np.float32))

            ct = np.concatenate([cal_idx, test_idx])
            std_ct = P1.model_uncertainty(
                args.model, model, X[train_idx].astype(np.float32), y[train_idx],
                X[ct].astype(np.float32), seed, args.n_members,
            )
            rf_std_cal, rf_std_test = std_ct[:len(cal_idx)], std_ct[len(cal_idx):]

            pred_cal, y_cal = pred[cal_idx], y[cal_idx]
            pred_test, y_test = pred[test_idx], y[test_idx]

            expo = CL.train_neighbor_exposure(test_idx, train_idx, neigh_idx, neigh_sim, y, pred, cfg)
            cliffB = expo["exposed"][test_idx]
            cliffA = cl["is_cliff"][test_idx]
            if cliffA.sum() < 5 or (~cliffA).sum() < 5:
                print(f"    [skip] cliffA degenerate {proto} seed={seed}", flush=True)
                continue

            st_cal = DP.neighbor_label_stats(
                fps, y, train_idx, cal_idx, neigh_idx,
                sim_floor=args.sim_floor, k=args.knn_stats_k,
                pair_sim=cfg.sim_threshold, pair_act=cfg.act_threshold)
            st_test = DP.neighbor_label_stats(
                fps, y, train_idx, test_idx, neigh_idx,
                sim_floor=args.sim_floor, k=args.knn_stats_k,
                pair_sim=cfg.sim_threshold, pair_act=cfg.act_threshold)

            methods = {"global": CP.global_intervals(pred_cal, y_cal, pred_test, args.alpha)}
            for mn, (gc, gt) in (("mondrian_wstd", (st_cal["d_wstd"], st_test["d_wstd"])),
                                 ("mondrian_rfstd", (rf_std_cal, rf_std_test)),
                                 ("mondrian_nb", (st_cal["d_nb"], st_test["d_nb"]))):
                lc, lt = P11.bins_by_cal(gc, gt, args.n_bins)
                methods[mn] = P1.mondrian_intervals(lc, pred_cal, y_cal, lt, pred_test,
                                                    args.alpha, args.min_cal)
            if getattr(args, "with_adaptive", False):
                r_fit = CP.oob_abs_residual(model, X[train_idx], y[train_idx])
                methods["adaptive"] = CP.adaptive_intervals(
                    X[train_idx], r_fit, X[cal_idx], pred_cal, y_cal, X[test_idx], pred_test,
                    args.alpha, seed=seed)

            nn_sim = max_sim_to_train(fps, train_idx, test_idx) if args.dump_sim else np.full(len(test_idx), np.nan)

            cov_all = {}
            for mname, iv in methods.items():
                cov = (y_test >= iv["lower"]) & (y_test <= iv["upper"])
                wid = iv["upper"] - iv["lower"]
                cov_all[mname] = cov
                m_rows.append(dict(
                    dataset=name, protocol=proto, seed=seed, method=mname,
                    n_test=len(test_idx), n_train=len(train_idx), n_cal=len(cal_idx),
                    n_cliffA=int(cliffA.sum()), n_cliffB=int(cliffB.sum()),
                    cov_all=float(cov.mean()),
                    cov_cliffA=float(cov[cliffA].mean()),
                    cov_noncliffA=float(cov[~cliffA].mean()),
                    gapA_pp=float(100 * (cov[~cliffA].mean() - cov[cliffA].mean())),
                    cov_cliffB=float(cov[cliffB].mean()) if cliffB.sum() else np.nan,
                    cov_noncliffB=float(cov[~cliffB].mean()),
                    gapB_pp=float(100 * (cov[~cliffB].mean() - cov[cliffB].mean())) if cliffB.sum() else np.nan,
                    width_all=float(wid.mean()),
                    width_cliffA=float(wid[cliffA].mean()),
                    nn_sim_mean=float(np.nanmean(nn_sim)),
                    nn_sim_median=float(np.nanmedian(nn_sim)),
                    y_mean_test=float(y_test.mean()), y_mean_train=float(y[train_idx].mean()),
                ))

            cg = cov_all["global"]
            for j, gi in enumerate(test_idx):
                molf_rows.append((name, proto, seed, int(gi), float(y_test[j]), float(pred_test[j]),
                                  bool(cliffA[j]), bool(cliffB[j]), bool(cg[j]),
                                  float(nn_sim[j]) if np.isfinite(nn_sim[j]) else np.nan))

            gl = m_rows[-len(methods)]
            print(f"    [{proto:12s} s={seed}] test={len(test_idx)} cliffA={int(cliffA.sum())} | "
                  f"cov all={gl['cov_all']:.3f} cliffA={gl['cov_cliffA']:.3f} "
                  f"gapA={gl['gapA_pp']:+.1f}pp | nn_sim_med={gl['nn_sim_median']:.3f} "
                  f"| dy={gl['y_mean_test'] - gl['y_mean_train']:+.2f}", flush=True)

    os.makedirs(os.path.join(out_root, "dump"), exist_ok=True)
    if molf_rows:
        pd.DataFrame(molf_rows, columns=["dataset", "protocol", "seed", "mol_idx", "y", "pred",
                                        "cliffA", "cliffB", "covered_global", "nn_sim_train"]) \
            .to_csv(os.path.join(out_root, "dump", f"{name}_spectrum.csv.gz"),
                    index=False, compression="gzip")
    return m_rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results/phase15_spectrum")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--protocols", default=",".join(PROTOCOLS))
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--sim", type=float, default=0.85)
    ap.add_argument("--act", type=float, default=1.0)
    ap.add_argument("--frac-train", type=float, default=0.6)
    ap.add_argument("--frac-cal", type=float, default=0.2)
    ap.add_argument("--frac-test", type=float, default=0.2)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--max-neighbors", type=int, default=256)
    ap.add_argument("--sim-floor", type=float, default=0.5)
    ap.add_argument("--knn-stats-k", type=int, default=50)
    ap.add_argument("--min-cal", type=int, default=30)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--model", default="rf")
    ap.add_argument("--n-members", type=int, default=6)
    ap.add_argument("--dump-sim", action="store_true", default=True)
    ap.add_argument("--no-dump-sim", dest="dump_sim", action="store_false")
    ap.add_argument("--with-adaptive", action="store_true",
                    help="额外计算 adaptive（归一化）共形臂——用于活性偏移下能否救回的对照")
    ap.add_argument("--min-size", type=int, default=800)
    ap.add_argument("--min-cliff", type=int, default=60)
    ap.add_argument("--max-datasets", type=int, default=15)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--seed0", type=int, default=20260923)
    args = ap.parse_args()

    args.protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    bad = [p for p in args.protocols if p not in PROTOCOLS]
    if bad:
        raise SystemExit(f"未知协议 {bad}；可选 {PROTOCOLS}")
    cfg = CL.CliffConfig(sim_threshold=args.sim, act_threshold=args.act)
    os.makedirs(args.out, exist_ok=True)

    files = find_moleculeace_datasets(args.data_dir)
    if args.datasets:
        wanted = [w.strip() for w in args.datasets.split(",") if w.strip()]
        files = [p for p in files if any(w in os.path.basename(p) for w in wanted)]
    print(f"[phase15] {len(files)} datasets; protocols={args.protocols}; seeds={args.seeds}", flush=True)

    all_m = []
    for p in files[: args.max_datasets]:
        name = os.path.basename(p).replace(".csv", "")
        print(f"=== {name}", flush=True)
        try:
            df = load_potency_csv(p, verbose=False)
            m, _ = run_one_dataset(df, name, cfg, args, args.out)
            if m:
                all_m.extend(m)
        except Exception:
            print(f"  [ERROR] {name}\n{traceback.format_exc()}", flush=True)
        if all_m:
            pd.DataFrame(all_m).to_csv(os.path.join(args.out, "METHODS.csv"), index=False)

    print(f"[phase15] DONE rows={len(all_m)}", flush=True)


if __name__ == "__main__":
    main()
