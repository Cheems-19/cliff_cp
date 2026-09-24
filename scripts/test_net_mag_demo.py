# -*- coding: utf-8 -*-
"""net_mag 工具的合成数据健全性检查（对齐论文机制）。

机制：信号在校准侧与分数耦合（低信号箱 → 校准分数低 → 分位低 → 收紧）；
悬崖组落在这个被收紧的箱里，但分数局部"破裂"（反常高）→ 失败集中。
预言：net_mag < 0 且真实 ΔCov(悬崖) < 0，符号应一致。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from cliffcp.diagnostics import net_mag, conformal_quantile, mondrian_quantiles

rng = np.random.default_rng(20260924)
n_cal, n_test = 4000, 4000
alpha = 0.10

# 校准侧：分数与信号耦合（b_cal 越大，信号越能解释难度）
cal_scores = rng.lognormal(mean=-2.0, sigma=1.0, size=n_cal)
cal_sig = rng.normal(size=n_cal) + 0.2 * rng.normal(size=n_cal)
q_bar = conformal_quantile(cal_scores, alpha)
print(f"q_bar = {q_bar:.4f}\n")

print(f"{'场景':22s} {'net_mag':>9s} {'ΔCov悬崖':>10s} {'ΔCov全体':>10s}  判定")
n_ok = 0

def run_case(tag, b_cal, cliff_where, lift, noise=0.3):
    """cliff_where: 'low'=信号判'易'的一块, 'high'=信号判'难'的一块(对齐情形)."""
    global edges, q_bins
    # 校准：低信号 → 小分数（信号如实解释校准难度），分箱后低信号箱分位低
    sc = rng.lognormal(mean=-2.0, sigma=1.0, size=n_cal) * np.exp(b_cal * cal_sig)
    edges, q_bins, _, qb = mondrian_quantiles(sc, cal_sig, n_bins=3, alpha=alpha)

    # 测试：非悬崖分子延续校准规律；悬崖组局部破裂（分数额外抬升 lift 倍）
    sig_test = rng.normal(size=n_test) + 0.2 * rng.normal(size=n_test)
    base = rng.lognormal(mean=-2.0, sigma=1.0, size=n_test) * np.exp(b_cal * sig_test)
    if cliff_where == "low":
        thr = np.quantile(sig_test, 0.10)
        cliff = sig_test <= thr
    else:
        thr = np.quantile(sig_test, 0.90)
        cliff = sig_test >= thr
    base = base * np.where(cliff, lift, 1.0) * np.exp(noise * rng.normal(size=n_test))
    sig_scores_test = base

    nm = net_mag(sig_test, sc, cal_sig, group=cliff, n_bins=3, alpha=alpha)
    b = np.searchsorted(edges, sig_test, side="right")
    cov_mond = sig_scores_test <= q_bins[b]
    cov_glob = sig_scores_test <= qb
    d_cliff = cov_mond[cliff].mean() - cov_glob[cliff].mean()
    d_all = cov_mond.mean() - cov_glob.mean()

    # 判定：符号一致，或两者都落在"无实质影响"区（|net_mag|<0.01 且 |ΔCov|<0.5pp）
    both_null = abs(nm) < 0.01 and abs(d_cliff) * 100 < 0.5
    ok = both_null or (np.sign(nm) == np.sign(d_cliff))
    verdict = "null(一致)" if both_null else ("OK" if ok else "MISMATCH")
    print(f"{tag:22s} {nm:>+9.3f} {d_cliff*100:>+9.2f}p {d_all*100:>+9.2f}p  {verdict}")
    return ok

n_ok += run_case("强错配(b=0.8,破1.6x)", 0.8, "low", 1.6)
n_ok += run_case("中错配(b=0.5,破1.3x)", 0.5, "low", 1.3)
n_ok += run_case("弱错配(b=0.3,破1.15x)", 0.3, "low", 1.15)
n_ok += run_case("对齐信号(悬崖=最难)", 0.8, "high", 1.0)   # 信号如实预警→分到大分位→不受伤
n_ok += run_case("无信息信号(b=0)", 0.0, "low", 1.0)        # 无耦合→net_mag≈0, ΔCov≈0

print(f"\n判定一致 {n_ok}/5")
print("[OK] diagnostics.py 健全性检查通过" if n_ok >= 4 else "[FAIL] 请检查")
