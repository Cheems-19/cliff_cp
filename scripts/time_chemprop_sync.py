#!/usr/bin/env python
"""chemprop v2 计时（同步执行，直接打印，不走管道避免缓冲吞输出）。

用 2 epochs 测单 epoch 成本，再乘以外推——比跑满 30 epochs 省时间得多。
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("CLIFFCP_ROOT", "."))
CP = ROOT / ".conda-chemprop" / "bin" / "chemprop"
PY = ROOT / ".conda-chemprop" / "bin" / "python"
CSV = Path("/tmp/cp_smoke.csv")


def run(label, args, timeout=1800):
    t0 = time.time()
    # 注意：chemprop 的第一个位置参数是子命令（train / predict / ...），
    # 漏掉它会把 --data-path 的值当成 mode，报 "invalid choice"。
    p = subprocess.run([str(CP), "train", *args], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    print(f"[{label}] exit={p.returncode}  elapsed={dt:.1f}s", flush=True)
    tail = (p.stdout or "")[-400:] + (p.stderr or "")[-400:]
    if tail.strip():
        print("   ---- tail ----")
        for line in tail.strip().splitlines()[-6:]:
            print("   ", line)
    return dt, p.returncode


# 数据准备
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402
from cliffcp import data as DA  # noqa: E402

dss = DA.find_moleculeace_datasets(str(ROOT / "data" / "MoleculeACE"))
pick = [p for p in dss if "CHEMBL233_Ki" in p] or dss[:1]
df = DA.load_potency_csv(pick[0], verbose=False)
pd.DataFrame({"smiles": df["smiles_canonical"].astype(str),
              "y": df["y"].astype(float)}).to_csv(CSV, index=False)
print(f"data: {CSV} rows={len(df)}", flush=True)

base = ["-i", str(CSV), "-t", "regression",
        "--target-columns", "y", "-s", "smiles",
        "--accelerator", "cpu", "--num-workers", "2"]

NE = 6          # 计时用的 epoch 数（必须 > warmup，chemprop 默认 warmup=2）
for label, extra, out in [
    ("A: 1 model x %d epochs" % NE,
     ["--ensemble-size", "1", "--epochs", str(NE), "--warmup-epochs", "1"], "/tmp/cp_a"),
    ("B: 3 models x %d epochs" % NE,
     ["--ensemble-size", "3", "--epochs", str(NE), "--warmup-epochs", "1"], "/tmp/cp_b"),
]:
    shutil.rmtree(out, ignore_errors=True)
    dt, rc = run(label, [*base, *extra, "-o", out, "--pytorch-seed", "1"])
    n_models = 1 if label.startswith("A") else 3
    per_epoch_per_model = dt / (NE * n_models)
    print(f"   -> 单模型单 epoch ≈ {per_epoch_per_model:.1f}s  "
          f"（30 epoch 单模型 ≈ {per_epoch_per_model*30/60:.1f} min）", flush=True)
    cks = sorted(Path(out).rglob("*.ckpt"))
    print(f"   -> 产物 ckpt 数 = {len(cks)}", flush=True)
    if cks:
        print(f"   -> 例：{cks[0].relative_to(out)}", flush=True)

# 预测计时
if Path("/tmp/cp_b").exists():
    cands = sorted(Path("/tmp/cp_b").rglob("*.ckpt"))
    if cands:
        models = [str(c) for c in cands]
        t0 = time.time()
        p = subprocess.run([str(CP), "predict", "-i", str(CSV),
                            "--model-paths", *models,
                            "-o", "/tmp/cp_preds.csv",
                            "--uncertainty-method", "ensemble",
                            "--accelerator", "cpu"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
        print(f"[C: predict ensemble] exit={p.returncode} elapsed={time.time()-t0:.1f}s", flush=True)
        tail = (p.stdout or "")[-500:] + (p.stderr or "")[-500:]
        for line in tail.strip().splitlines()[-8:]:
            print("   ", line)
        pf = Path("/tmp/cp_preds.csv")
        if pf.exists():
            print("   ---- preds head ----", flush=True)
            print("   " + pf.read_text().splitlines()[0][:300], flush=True)
            print("   " + pf.read_text().splitlines()[1][:300], flush=True)
