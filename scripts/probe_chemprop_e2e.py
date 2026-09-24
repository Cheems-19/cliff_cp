#!/usr/bin/env python
"""Chemprop D-MPNN 端到端可行性验证（单数据集单划分，便宜版）。

要回答四个问题，全都用数据回答，不靠"应该能行"：
  1. 训练能否产出可用 checkpoint？（chemprop 2.3.1 在写 CSV 日志时会抛
     `dict contains fields not in fieldnames: 'train_loss_step'`，但 best-*.ckpt 已经落盘
     —— 所以按"checkpoint 存在即视为成功"处理，并把这一步显式写出来）
  2. checkpoint 能否被 MPNN.load_from_checkpoint 加载？
  3. 预测是否合理？（预测值与真实值的 RMSE / 相关系数；与 RF 的量级对比）
  4. 模型侧不确定性是否非退化？（分布宽度、与 |残差| 的相关性 —— 若方差≈0 就是假信号）
  另外顺手算一次分裂共形覆盖率，确认整条链能接上。

用法: python scripts/probe_chemprop_e2e.py --dataset CHEMBL233_Ki --epochs 10
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CLIFFCP_ROOT", "."))
CP = ROOT / ".conda-chemprop" / "bin" / "chemprop"
TMP = Path("/tmp/cp_e2e")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CHEMBL233_Ki")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--frac-train", type=float, default=0.6)
    ap.add_argument("--frac-cal", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--accelerator", default="cuda",
                    help="cuda 或 cpu。GPU0 通常有 ~10G 空闲，D-MPNN 占用很小；"
                         "CPU 上实测 30–130 s/epoch，30 epoch 要 15–65 min，太慢。")
    ap.add_argument("--gpu", default="0", help="GPU 序号（--devices 的值）")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from cliffcp import data as DA

    dss = DA.find_moleculeace_datasets(str(ROOT / "data" / "MoleculeACE"))
    pick = [p for p in dss if args.dataset in p]
    if not pick:
        raise SystemExit(f"找不到数据集 {args.dataset}")
    df = DA.load_potency_csv(pick[0], verbose=False)
    n = len(df)
    print(f"[data] {args.dataset} n={n}", flush=True)

    # 自己切 train/cal/test，用 chemprop 的 split 列（0=train,1=val,2=test）
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_tr = int(round(args.frac_train * n))
    n_ca = int(round(args.frac_cal * n))
    split = np.full(n, 2, dtype=int)
    split[perm[:n_tr]] = 0
    split[perm[n_tr:n_tr + n_ca]] = 1     # 用校准集当 chemprop 的 val（反正我们要自己算 CP）

    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    allp = TMP / "all.csv"
    # 坑：--splits-column 必须是**字符串**标签（train/val/test）。
    # 给整数会让 chemprop 内部对列做 .str 访问，直接
    # AttributeError: Can only use .str accessor with string values, not integer
    split_lab = np.array(["test"] * n, dtype=object)
    split_lab[perm[:n_tr]] = "train"
    split_lab[perm[n_tr:n_tr + n_ca]] = "val"
    pd.DataFrame({
        "smiles": df["smiles_canonical"].astype(str),
        "y": df["y"].astype(float),
        "split": split_lab,
    }).to_csv(allp, index=False)
    calp = TMP / "cal.csv"
    pd.DataFrame({
        "smiles": df["smiles_canonical"].astype(str),
        "y": df["y"].astype(float),
    }).iloc[perm[n_tr:n_tr + n_ca]].to_csv(calp, index=False)
    print(f"[split] train={n_tr} cal={n_ca} test={n - n_tr - n_ca}", flush=True)

    out = TMP / "train"
    cmd = [str(CP), "train", "-i", str(allp), "-o", str(out),
           "-t", "regression", "--target-columns", "y", "-s", "smiles",
           "--splits-column", "split",
           "--epochs", str(args.epochs), "--warmup-epochs", "1",
           "--patience", "4", "--ensemble-size", "1",
           "--accelerator", args.accelerator,
           "--num-workers", "2",
           "--pytorch-seed", str(args.seed), "--data-seed", str(args.seed),
           "--remove-checkpoints"]
    t0 = time.time()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu}
    p = subprocess.run(cmd, cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=7200)
    dt = time.time() - t0
    print(f"[train] exit={p.returncode} elapsed={dt:.0f}s "
          f"({dt/args.epochs:.1f}s/epoch over {args.epochs} epochs)", flush=True)
    if p.returncode != 0:
        err = (p.stderr or "")[-600:]
        print("[train] 非零退出，最后几行 stderr：", flush=True)
        for line in err.strip().splitlines()[-6:]:
            print("   ", line, flush=True)
        print("[train] 说明：chemprop 2.3.1 的 CSVLogger 在新版 lightning 下会抛 "
              "'dict contains fields not in fieldnames' —— 只要 checkpoint 已落盘即可继续", flush=True)

    # 坑：--remove-checkpoints 之后 chemprop 保存的是 **best.pt**（Lightning .pt），
    # 不是 best-*.ckpt。只按 *.ckpt 找会得到"0 个 checkpoint"并误判为失败。
    cks = (sorted(out.rglob("best*.pt")) or sorted(out.rglob("*.pt"))
           or sorted(out.rglob("best-*.ckpt")) or sorted(out.rglob("*.ckpt")))
    print(f"[ckpt] 找到 {len(cks)} 个: {[c.name for c in cks]}", flush=True)
    if not cks:
        print("!! 没有 checkpoint，无法继续", flush=True)
        return

    # 预测（用 ensemble 模式；单模型时等价于直接预测）
    predf = TMP / "preds.csv"
    pc = [str(CP), "predict", "-i", str(allp), "--model-paths", *[str(c) for c in cks],
          "-o", str(predf), "--accelerator", args.accelerator,
          "--uncertainty-method", "ensemble"]
    t1 = time.time()
    env_p = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu}
    pp = subprocess.run(pc, cwd=str(ROOT), env=env_p,
                        capture_output=True, text=True, timeout=3600)
    print(f"[predict] exit={pp.returncode} elapsed={time.time()-t1:.0f}s", flush=True)
    if pp.returncode != 0 or not predf.exists():
        for line in ((pp.stderr or "")[-500:]).strip().splitlines()[-6:]:
            print("   ", line, flush=True)
        # 退路：chemprop 训练时自己写了 test_predictions.csv
        fallback = out / "model_0" / "test_predictions.csv"
        if fallback.exists():
            print(f"   退路：改用 chemprop 自带的 {fallback.name}", flush=True)
            predf = fallback
        else:
            return

    P = pd.read_csv(predf)
    print(f"[preds] columns = {list(P.columns)[:12]}", flush=True)
    cols = list(P.columns)
    low = {c.lower(): c for c in cols}
    ycol = next((low[k] for k in ("y", "target", "true", "y_true", "activity") if k in low),
                cols[0])
    pcol = next((c for c in cols if c.lower().startswith("pred")), None)
    if pcol is None:
        pcol = [c for c in cols if c not in (ycol,)][0]
    pv = P[pcol].to_numpy(float)
    yv = P[ycol].to_numpy(float)
    un = [c for c in cols if "unc" in c.lower() or "var" in c.lower()]
    uv = P[un[0]].to_numpy(float) if un else np.full(len(P), np.nan)
    print(f"[preds] 判定列名：预测={pcol} 真值={ycol} "
          f"不确定性={un[0] if un else '(无)'} n={len(P)}", flush=True)

    print(f"[preds] RMSE(all) = {np.sqrt(np.mean((pv-yv)**2)):.4f} | "
          f"corr = {np.corrcoef(pv, yv)[0,1]:.4f}", flush=True)
    if np.isfinite(uv).any():
        print(f"[unc] mean={np.nanmean(uv):.4f} sd={np.nanstd(uv):.4f} "
              f"min={np.nanmin(uv):.4f} max={np.nanmax(uv):.4f}", flush=True)
        rc = np.abs(yv - pv)
        m = np.isfinite(uv)
        print(f"[unc] corr(unc, |residual|) = {np.corrcoef(uv[m], rc[m])[0,1]:.4f}",
              flush=True)
        print("       （≈0 说明该不确定性是假信号，不足以当模型侧对手）", flush=True)

    # 分裂共形覆盖率
    te = split == 2
    ca = split == 1
    sc = np.abs(yv[ca] - pv[ca])
    k = int(np.ceil((len(sc) + 1) * (1 - args.alpha)))
    q = np.sort(sc)[min(k, len(sc)) - 1]
    cov = np.mean(np.abs(yv[te] - pv[te]) <= q)
    print(f"[conformal] alpha={args.alpha} q={q:.4f} "
          f"test coverage={cov:.4f} (nominal {1-args.alpha})", flush=True)
    print(f"[done] 端到端可行", flush=True)


if __name__ == "__main__":
    main()
