# -*- coding: utf-8 -*-
"""B2：net_mag 的统计强度（置换检验 + bootstrap CI + 留一 + 信号级检验）"""
import io, os
import numpy as np
import pandas as pd
from scipy import stats

C = "C:/Users/Administrator/Desktop/小论文/cliff_cp"
H = pd.read_csv(C + "/results_phase14/TRANSFER_HARM.csv")
a = H.groupby(["run", "signal"]).mean(numeric_only=True).reset_index()
print("单元数 n =", len(a), "| 信号数 =", a.signal.nunique(), "| run 数 =", a.run.nunique())

x = a.net_mag.to_numpy(float)
y = a.d_cov_cliff.to_numpy(float)

rho = stats.spearmanr(x, y).statistic
r = np.corrcoef(x, y)[0, 1]
print(f"\n[主结果] Spearman rho = {rho:+.3f}   Pearson r = {r:+.3f}   (n={len(x)})")

# 1) 置换检验（打乱 d_cov 的对应关系）
rng = np.random.default_rng(0)
B = 200000
null = np.empty(B)
idx = np.arange(len(y))
for b in range(B):
    p = rng.permutation(idx)
    null[b] = stats.spearmanr(x, y[p]).statistic
p_perm = float((np.abs(null) >= abs(rho)).mean())
print(f"[置换检验] 双尾 p = {p_perm:.4g}  (B={B})；零分布 |rho| 95% 分位 = {np.quantile(np.abs(null), .95):.3f}")

# 2) bootstrap CI（重抽样 16 个单元）
bs = np.empty(10000)
for b in range(10000):
    i = rng.integers(0, len(y), len(y))
    if len(set(i)) < 4:
        bs[b] = np.nan; continue
    bs[b] = stats.spearmanr(x[i], y[i]).statistic
bs = bs[np.isfinite(bs)]
lo, hi = np.quantile(bs, [.025, .975])
print(f"[bootstrap] rho 95% CI = [{lo:+.3f}, {hi:+.3f}]")

# 3) 留一（drop-one-unit）
loo = []
for i in range(len(y)):
    m = np.ones(len(y), bool); m[i] = False
    loo.append(stats.spearmanr(x[m], y[m]).statistic)
loo = np.array(loo)
print(f"[留一] rho 范围 = [{loo.min():+.3f}, {loo.max():+.3f}]；剔除后仍 >0.7 的比例 = "
      f"{(loo > 0.7).mean():.0%}")
worst = a.iloc[int(np.argmin(loo))]
print(f"       影响最大的单元：{worst.run} / {worst.signal}（剔除后 rho={loo.min():+.3f}）")

# 4) 信号级检验（每个信号在两个 run 上取均值 -> n=8，消除表示轴的相关）
s = a.groupby("signal").mean(numeric_only=True).reset_index()
xs, ys = s.net_mag.to_numpy(float), s.d_cov_cliff.to_numpy(float)
rho_s = stats.spearmanr(xs, ys).statistic
null_s = np.array([stats.spearmanr(xs, ys[rng.permutation(len(ys))]).statistic for _ in range(50000)])
print(f"\n[信号级 n=8] Spearman rho = {rho_s:+.3f}, 置换 p = {float((np.abs(null_s)>=abs(rho_s)).mean()):.4g}")
print(s[["signal", "net_mag", "d_cov_cliff"]].to_string(index=False))

# 5) 与「裸错配量 M」对比：M 是否更差？（用已知 4 信号的 M 值做定性说明）
print("\n[对照] 单元级 |rho| 的稳健性：")
print(f"  最大杠杆点 = {np.max(np.abs(x - x.mean()))/x.std():.2f} 个标准差")

# 6) AD 分箱表（供 B5 的「非同义反复」论证）
print("\n=== AD_BY_SIMBIN.csv（同相似度层内的差异）===")
S = pd.read_csv(C + "/results_ecfp_abl/AD_BY_SIMBIN.csv")
print(S.to_string(index=False))
print("\n=== AD_2x2.csv ===")
T = pd.read_csv(C + "/results_ecfp_abl/AD_2x2.csv")
print(T.to_string(index=False))

out = []
def w(*z): out.append(" ".join(str(q) for q in z))
w("=== B2 net_mag 统计强度 ===")
w(f"n_units={len(x)}  Spearman={rho:+.3f}  Pearson={r:+.3f}")
w(f"permutation p={p_perm:.4g} (B={B})")
w(f"bootstrap 95% CI=[{lo:+.3f},{hi:+.3f}]")
w(f"leave-one-out range=[{loo.min():+.3f},{loo.max():+.3f}]")
w(f"signal-level n=8 Spearman={rho_s:+.3f} perm_p={float((np.abs(null_s)>=abs(rho_s)).mean()):.4g}")
io.open(C + "/results_phase14/STAT_STRENGTH.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n[saved] results_phase14/STAT_STRENGTH.txt")
