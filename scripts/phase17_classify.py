# -*- coding: utf-8 -*-
"""B6：外部效度 —— 把「悬崖邻域覆盖失效」外推到分类任务（集合预测共形）。

数据：MoleculeNet 的 BBBP / BACE / Tox21(NR-AR)，首次运行自动从 DeepChem S3 下载缓存到
     data/classification/（服务器有外网）。
设计（与回归主实验同构）：
  - 特征 ECFP4(2048)，RF 分类器 200 棵树（与回归主实验同模型族）；
  - 随机划分 train/cal/test = 60/20/20，多种子；
  - 共形集合预测：score = 1 − p̂(y)，q̂ = conformal_quantile(alpha)；
    集合 = {k: p̂(k) ≥ 1 − q̂}；覆盖 = 真标签 ∈ 集合。
  - 「悬崖」类比：**标签悬崖** —— 测试分子存在训练邻居（Tanimoto ≥ sim）且标签不一致
    （回归里是标签差大；分类里是标签冲突，即稠密邻域里的标签矛盾）。
  - 条件化对手：Mondrian ← 预测类（可部署的类条件化）；Mondrian ← nn_sim（距离型，与回归同约定）。
指标：覆盖（全部 / 标签悬崖 / 非悬崖）、集合大小、按真实类别的覆盖。
"""
from __future__ import annotations

import os
import sys
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cliffcp import features as F  # noqa: E402
from cliffcp import stats as S  # noqa: E402
from cliffcp.conformal import conformal_quantile  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data", "classification")
S3 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
URL_MAP = {"BBBP.csv": "BBBP.csv", "BACE.csv": "bace.csv", "tox21.csv": "tox21.csv.gz",
           "HIV.csv": "HIV.csv", "ClinTox.csv": "clintox.csv.gz"}
SIM = 0.85
ALPHA = 0.10
MIN_CAL = 30


def download(name: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    p = os.path.join(DATA_DIR, name)
    if not os.path.exists(p):
        url = S3 + URL_MAP[name]
        print(f"[download] {url}", flush=True)
        import subprocess
        r = subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", p, url],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not os.path.exists(p):
            raise RuntimeError(f"curl 下载失败: {r.stderr[-300:]}")
    return p


def load_datasets():
    out = {}
    # BBBP: smiles, p_np
    df = pd.read_csv(download("BBBP.csv"))
    out["BBBP"] = df.rename(columns={"smiles": "smi", "p_np": "y"})[["smi", "y"]].dropna()
    # BACE: mol/Class（MoleculeNet 的 bace.csv 用 mol 列存 SMILES）
    df = pd.read_csv(download("BACE.csv"))
    smi_col = "mol" if "mol" in df.columns else "smiles"
    out["BACE"] = df.rename(columns={smi_col: "smi", "Class": "y"})[["smi", "y"]].dropna()
    # Tox21: smiles + 12 tasks; 取 NR-AR，丢弃无标签（缓存文件是 gz 内容、.csv 名）
    df = pd.read_csv(download("tox21.csv"), compression="gzip")
    sub = df[["smiles", "NR-AR"]].dropna()
    out["Tox21_NR-AR"] = sub.rename(columns={"smiles": "smi", "NR-AR": "y"})
    # HIV: smiles, HIV_active
    df = pd.read_csv(download("HIV.csv"))
    out["HIV"] = df.rename(columns={"smiles": "smi", "HIV_active": "y"})[["smi", "y"]].dropna()
    # ClinTox: smiles 或 mol, CT_TOX（gz）
    df = pd.read_csv(download("ClinTox.csv"), compression="gzip")
    smi_col = "smiles" if "smiles" in df.columns else "mol"
    out["ClinTox"] = df.rename(columns={smi_col: "smi", "CT_TOX": "y"})[["smi", "y"]].dropna()
    # 规范化 + 去重（同分子取标签 max）
    for name, d in out.items():
        d = d.copy()
        d["smi"] = [F.canonicalize(s) or "" for s in d["smi"]]
        d = d[d.smi != ""]
        d["y"] = d["y"].astype(int)
        d = d.groupby("smi", as_index=False).agg(y=("y", "max")).reset_index(drop=True)
        out[name] = d
        print(f"[data] {name}: {len(d)} 分子（正类 {d.y.mean():.1%}）", flush=True)
    return out


def conformal_sets(proba_test, qhat):
    """集合 = {k: p̂(k) ≥ 1 − q̂}；qhat 可为标量或逐分子数组；返回布尔矩阵 (n, K)。"""
    thr = np.asarray(qhat, float)
    if thr.ndim == 1:
        thr = thr[:, None]
    return proba_test >= (1.0 - thr)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase17_classify")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--sim", type=float, default=SIM)
    ap.add_argument("--datasets", default="BBBP,BACE,Tox21_NR-AR")
    ap.add_argument("--seed0", type=int, default=20260923)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "dump"), exist_ok=True)
    wanted = args.datasets.split(",")

    rows = []
    for name in wanted:
        d = load_datasets()[name]
        smiles = d.smi.tolist()
        y = d.y.to_numpy()
        n = len(y)
        fps, keep = F.morgan_fps(smiles)
        keep = np.asarray(keep, int)
        smiles = [smiles[i] for i in keep]
        y = y[keep]
        n = len(y)
        X = F.fps_to_matrix(fps)
        neigh_idx, neigh_sim = F.tanimoto_neighbors(fps, sim_threshold=args.sim)
        print(f"[{name}] 邻居计算完成", flush=True)

        for si in range(args.seeds):
            seed = args.seed0 + si * 7919
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n)
            n_tr = int(round(0.6 * n))
            n_ca = int(round(0.2 * n))
            train_idx, cal_idx, test_idx = idx[:n_tr], idx[n_tr:n_tr + n_ca], idx[n_tr + n_ca:]

            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=8)
            clf.fit(X[train_idx], y[train_idx])
            proba_cal = clf.predict_proba(X[cal_idx])
            proba_test = clf.predict_proba(X[test_idx])
            classes = clf.classes_
            k_of = {c: j for j, c in enumerate(classes)}

            # 标签悬崖：存在训练邻居且标签不一致
            y_train = y[train_idx]
            pos = {m: k for k, m in enumerate(train_idx)}
            lc = np.zeros(len(test_idx), bool)
            for j, ti in enumerate(test_idx):
                nb = neigh_idx[ti]
                if len(nb) == 0:
                    continue
                nb_labels = np.array([y_train[pos[m]] for m in nb if m in pos], dtype=int)
                if len(nb_labels) and (nb_labels != y[ti]).any():
                    lc[j] = True

            # 全局 CP
            p_true_cal = np.array([proba_cal[r, k_of[c]] for r, c in enumerate(y[cal_idx])])
            scores = 1.0 - p_true_cal
            qg = conformal_quantile(scores, args.alpha)
            p_true_test = np.array([proba_test[r, k_of[c]] for r, c in enumerate(y[test_idx])])
            set_g = conformal_sets(proba_test, qg)
            cov_g = set_g[np.arange(len(test_idx)), [k_of[c] for c in y[test_idx]]]
            size_g = set_g.sum(axis=1)

            def mondrian(sig_cal, sig_test):
                edges = S.bin_edges_from(sig_cal, 5)
                bc = S.assign_bins(sig_cal, edges)
                bt = S.assign_bins(sig_test, edges)
                q = np.full(len(eps_t := sig_test), np.nan)  # noqa: F841
                qm = {}
                for bb in np.unique(bc):
                    m = bc == bb
                    qm[bb] = conformal_quantile(scores[m], args.alpha) if m.sum() >= MIN_CAL else qg
                qt = np.array([qm.get(b, qg) for b in bt], float)
                st = conformal_sets(proba_test, qt)
                cov = st[np.arange(len(test_idx)), [k_of[c] for c in y[test_idx]]]
                return cov, st.sum(axis=1), qt

            pred_class_cal = classes[np.argmax(proba_cal, axis=1)]
            pred_class_test = classes[np.argmax(proba_test, axis=1)]
            cov_pc, size_pc, _ = mondrian(
                (pred_class_cal == 1).astype(float), (pred_class_test == 1).astype(float))

            # nn_sim 信号（到训练集最大相似度；无邻居 NaN → 与回归同约定落顶箱）
            nn_sim_test = np.array([neigh_sim[ti].max() if len(neigh_idx[ti]) else np.nan
                                    for ti in test_idx])
            nn_sim_cal = np.array([neigh_sim[ti].max() if len(neigh_idx[ti]) else np.nan
                                   for ti in cal_idx])
            cov_ns, size_ns, _ = mondrian(nn_sim_cal, nn_sim_test)

            # APS（Adaptive Prediction Sets，分类的自适应基线；确定性版本，趋保守）
            order_cal = np.argsort(-proba_cal, axis=1)
            cum_cal = np.take_along_axis(proba_cal, order_cal, axis=1).cumsum(axis=1)
            pos_true_cal = np.array([int(np.where(order_cal[r] == c)[0][0])
                                     for r, c in enumerate(y[cal_idx])])
            s_aps = cum_cal[np.arange(len(cal_idx)), pos_true_cal]
            q_aps = conformal_quantile(s_aps, args.alpha)
            order_t = np.argsort(-proba_test, axis=1)
            cum_t = np.take_along_axis(proba_test, order_t, axis=1).cumsum(axis=1)
            incl_sorted = cum_t <= q_aps
            incl_sorted[:, 0] = True                      # 保证集合非空
            set_aps = np.zeros_like(incl_sorted)
            np.put_along_axis(set_aps, order_t, incl_sorted, axis=1)
            cov_aps = set_aps[np.arange(len(test_idx)), [k_of[c] for c in y[test_idx]]]
            size_aps = set_aps.sum(axis=1)

            for mname, cov, size in [("global", cov_g, size_g),
                                     ("aps", cov_aps, size_aps),
                                     ("mondrian_predclass", cov_pc, size_pc),
                                     ("mondrian_nnsim", cov_ns, size_ns)]:
                rows.append(dict(
                    dataset=name, seed=seed, method=mname,
                    n_test=len(test_idx), n_cliff=int(lc.sum()),
                    cov_all=float(cov.mean()),
                    cov_cliff=float(cov[lc].mean()) if lc.sum() else np.nan,
                    cov_noncliff=float(cov[~lc].mean()) if (~lc).sum() else np.nan,
                    size_all=float(size.mean()),
                    size_cliff=float(size[lc].mean()) if lc.sum() else np.nan,
                    size_noncliff=float(size[~lc].mean()) if (~lc).sum() else np.nan,
                    q_global=qg,
                ))
            # 真值类别条件覆盖（global，报告用）
            for cval in [0, 1]:
                m = y[test_idx] == cval
                if m.sum():
                    rows[-3].setdefault(f"cov_global_class{cval}", float(cov_g[m].mean()))
            print(f"  [{name} s={seed}] cliff={int(lc.sum())} "
                  f"global cov={cov_g.mean():.3f} cliff={cov_g[lc].mean() if lc.sum() else float('nan'):.3f}",
                  flush=True)

    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(args.out, "METHODS.csv"), index=False)

    # 汇总
    L = ["=== B6 分类外推（α=0.10, 标签悬崖 = 训练邻居标签冲突）==="]
    for name, s in M.groupby("dataset"):
        L.append(f"\n── {name} (n_test/seed={int(s.n_test.iloc[0])}, 悬崖/seed≈{int(s.n_cliff.max())}) ──")
        piv = s.pivot_table(index="seed", columns="method",
                            values=["cov_all", "cov_cliff", "cov_noncliff", "size_all"],
                            aggfunc="mean")
        for meth in ["global", "aps", "mondrian_predclass", "mondrian_nnsim"]:
            try:
                cc = piv[("cov_cliff", meth)]
                cn = piv[("cov_noncliff", meth)]
            except KeyError:
                continue
            per = 100 * (cn - cc)
            gap = per.mean()
            pos = int((per > 0).sum())
            try:
                wl = stats.wilcoxon(per.dropna(), zero_method="wilcox").pvalue
            except Exception:
                wl = float("nan")
            L.append(f"  {meth:20s} cov_all={piv[('cov_all', meth)].mean():.3f} "
                     f"cliff={cc.mean():.3f} non={cn.mean():.3f} "
                     f"gap={gap:+.1f}pp ({pos}/{len(per)}, p={wl:.3g}) "
                     f"size={piv[('size_all', meth)].mean():.2f}")
    txt = "\n".join(L)
    with open(os.path.join(args.out, "SUMMARY.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
