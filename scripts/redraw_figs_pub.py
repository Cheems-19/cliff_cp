#!/usr/bin/env python
"""出版质量重绘脚本：Fig 2-8（Fig 9/10 已另有脚本，不在此处重绘）。

统一风格：
  - 300 dpi，DejaVu Sans（全英文标注，避免 CJK 字体缺失）
  - 面板字母 A/B/C 左上角加粗
  - 颜色语义一致：红 = 有害 / 蓝 = 改善 / 绿 = 对齐信号（标签离散度）/ 灰 = 中性
  - 去掉 top/right spine，统一字号

所有数据相对项目根目录；脚本自动 chdir 到根。
输出：figures/fig{2..8}_*.png
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

# ---------------- 统一出版风格 ----------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
})

# 颜色语义
HARM = "#b2182b"       # 红：有害（把覆盖从悬崖组搬走）
BETTER = "#2166ac"     # 蓝：改善悬崖组
DISP = "#1b7837"       # 绿：对齐信号（标签离散度）
NEUTRAL = "#8c8c8c"    # 灰：中性
REF = "#333333"        # 参考线
ORANGE = "#ef8a62"     # 密度/相似度侧（也是有害，但区别于模型侧）
PURPLE = "#762a83"     # 邻域矛盾侧


def letter(ax, ch, x=-0.14, y=1.03):
    ax.text(x, y, ch, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left", color="#111111")


def sig(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ============================================================
# Fig 2 — 机制：分区信号能否"看见"悬崖、条件化是否砸塌悬崖组
# ============================================================
def fig2():
    rf = pd.read_csv("results_phase2/E_cross_rfstd.csv")
    ws = pd.read_csv("results_phase2/F_cross_wstd.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    x = rf["bin"].to_numpy()

    ax = axes[0]
    ax.plot(x, rf["frac_cliffB"] * 100, "o-", color=HARM, label="binned by rf_std (model-side)",
            markeredgecolor="white", zorder=3)
    ax.plot(x, ws["frac_cliffB"] * 100, "s-", color=DISP, label="binned by d_wstd (label dispersion)",
            markeredgecolor="white", zorder=3)
    base = rf["n_cliffB"].sum() / rf["n"].sum() * 100
    ax.axhline(base, ls="--", c=NEUTRAL, lw=1, label=f"overall cliff share {base:.1f}%")
    ax.set_xticks(x)
    ax.set_xlabel("signal bin  (0 = smallest  →  4 = largest)")
    ax.set_ylabel("cliff share within bin (%)")
    ax.set_title("Does the signal rank activity cliffs?", fontsize=9.5)
    ax.legend(fontsize=7.2, loc="upper left", framealpha=0.9)
    letter(ax, "A")

    ax = axes[1]
    ax.plot(x, rf["cov_cliffB_global"], "o--", color=BETTER, label="global CP",
            markeredgecolor="white", zorder=3)
    ax.plot(x, rf["cov_cliffB_rfstd"], "o-", color=HARM, label="conditioned on rf_std",
            markeredgecolor="white", zorder=3)
    ax.plot(x, ws["cov_cliffB_wstd"], "s-", color=DISP, label="conditioned on d_wstd",
            markeredgecolor="white", zorder=3)
    ax.axhline(0.9, ls=":", c=NEUTRAL, lw=1.2, label="nominal 0.90")
    ax.set_xticks(x)
    ax.set_xlabel("signal bin  (0 = smallest  →  4 = largest)")
    ax.set_ylabel("cliff-group coverage within bin")
    ax.set_title("Which partition breaks the cliff group?", fontsize=9.5)
    ax.set_ylim(0.2, 1.0)
    ax.legend(fontsize=7.2, loc="lower right", framealpha=0.9)
    letter(ax, "B")

    fig.suptitle(
        "Cliff-group coverage: 0.723 (global) → 0.669 (rf_std) vs 0.742 (d_wstd)  ·  "
        "15 ChEMBL assays × 10 splits", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/fig2_mechanism.png")
    plt.close(fig)
    print("[saved] figures/fig2_mechanism.png")


# ============================================================
# Fig 3 — 分区变量消融：收益 vs 代价散点
# ============================================================
def fig3():
    d = pd.read_csv("results_ablation/PHASE1_ABLATION.csv")
    FAMILY = {
        "mondrian_rfstd": ("model-side (rf variance)", HARM),
        "mondrian_nb": ("density-side (n neighbours)", ORANGE),
        "mondrian_nnsim": ("similarity-side (NN sim)", ORANGE),
        "mondrian_pair": ("neighbourhood-conflict", PURPLE),
        "mondrian_pairgap": ("neighbourhood-conflict", PURPLE),
        "mondrian_iqr": ("label-dispersion family", DISP),
        "mondrian_std": ("label-dispersion family", DISP),
        "mondrian_wstd": ("label-dispersion family", DISP),
    }
    d["family"] = d["method"].map(lambda m: FAMILY.get(m, ("other", NEUTRAL))[0])
    d["color"] = d["method"].map(lambda m: FAMILY.get(m, ("other", NEUTRAL))[1])
    d["label"] = d["method"].str.replace("mondrian_", "", regex=False)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for fam, sub in d.groupby("family"):
        ax.scatter(sub["width_delta"], sub["delta_mean"], s=110,
                   c=sub["color"].iloc[0], label=fam, zorder=3,
                   edgecolors="white", linewidths=1.2)
    for _, r in d.iterrows():
        if r["label"] in ("std", "wstd", "pair", "iqr"):
            continue  # 近邻点改用引线标注（见下）
        dx, dy = {"pairgap": (10, -16), "nnsim": (10, -16)}.get(r["label"], (9, 6))
        ax.annotate(r["label"], (r["width_delta"], r["delta_mean"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=9)
    row_w = d.loc[d["label"] == "wstd"].iloc[0]
    row_s = d.loc[d["label"] == "std"].iloc[0]
    row_p = d.loc[d["label"] == "pair"].iloc[0]
    row_i = d.loc[d["label"] == "iqr"].iloc[0]
    leader = dict(arrowstyle="-", lw=0.7, color="#555555")
    ax.annotate("wstd", xy=(row_w["width_delta"], row_w["delta_mean"]),
                xytext=(row_w["width_delta"] - 0.018, row_w["delta_mean"] + 0.0052),
                fontsize=9, arrowprops=leader, zorder=6)
    ax.annotate("std", xy=(row_s["width_delta"], row_s["delta_mean"]),
                xytext=(row_s["width_delta"] + 0.016, row_s["delta_mean"] + 0.0038),
                fontsize=9, arrowprops=leader, zorder=6)
    ax.annotate("pair", xy=(row_p["width_delta"], row_p["delta_mean"]),
                xytext=(row_p["width_delta"] - 0.017, row_p["delta_mean"] - 0.0062),
                fontsize=9, arrowprops=leader, zorder=6)
    ax.annotate("iqr", xy=(row_i["width_delta"], row_i["delta_mean"]),
                xytext=(row_i["width_delta"] + 0.010, row_i["delta_mean"] + 0.0056),
                fontsize=9, arrowprops=leader, zorder=6)
    ax.axhline(0, ls="--", c=REF, lw=1)
    ax.axvline(0, ls="--", c=REF, lw=1)
    ax.set_xlim(d["width_delta"].min() - 0.012, d["width_delta"].max() + 0.012)
    ax.set_ylim(d["delta_mean"].min() - 0.008, d["delta_mean"].max() + 0.005)
    ax.set_xlabel("average interval-width change  (cost; negative = cheaper)")
    ax.set_ylabel("cliff-group coverage change\n(benefit; positive = safer)")
    ax.set_title("Which partition variable should you use?\ntop-left = better coverage AND cheaper intervals", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig("figures/fig3_ablation.png")
    plt.close(fig)
    print("[saved] figures/fig3_ablation.png")


# ============================================================
# Fig 4 — 代价-收益前沿：原始视图 vs 匹配宽度视图
# ============================================================
def fig4():
    G = pd.read_csv("results_ecfp_abl/FRONTIER_GAMMA.csv")
    P = pd.read_csv("results_ecfp_abl/FRONTIER_PAIRED.csv").set_index("rule")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.55, 1.3]})

    ax = axes[0]
    x = G["w_ratio"].to_numpy(float) * 100 - 100
    y = G["d_cliff_raw"].to_numpy(float) * 100
    ax.plot(x, y, "o-", color=HARM, lw=2.0, ms=6, markeredgecolor="white",
            label="one-sided rule  max(q_bin, γ·q_global)")
    for gv, xi, yi in zip(G["gamma"], x, y):
        if gv in (0.0, 0.8, 1.0):
            off = (-40, -6) if gv == 0.8 else (6, -12)  # 0.8 标注在点左侧，避让右下图例
            ax.annotate(f"γ={gv:.1f}", (xi, yi), textcoords="offset points",
                        xytext=off, fontsize=8, color=HARM)
    ax.axhline(0, ls="--", c=REF, lw=1.1)
    ax.axvline(0, ls="--", c=REF, lw=1.1)
    others = {
        "rfstd": ("model-side var. (misaligned)", HARM, "s"),
        "nb": ("# training neighbours", ORANGE, "^"),
        "wstd": ("label dispersion (aligned)", DISP, "D"),
        "std": ("label std", DISP, "v"),
        "iqr": ("label IQR", NEUTRAL, "P"),
        "nnsim": ("NN similarity", ORANGE, "X"),
    }
    for tag, (lab, col, mk) in others.items():
        if tag not in P.index:
            continue
        ax.scatter([P.loc[tag, "w_ratio"] * 100 - 100], [P.loc[tag, "d_cliff_raw"] * 100],
                   marker=mk, s=90, color=col, edgecolor="white", zorder=5,
                   label=lab if tag in ("rfstd", "wstd", "nb", "nnsim") else None)
    ax.set_xlabel("mean interval width vs global (%)")
    ax.set_ylabel("cliff-group coverage vs global (pp)")
    ax.set_title("Raw view: any gain tracks interval width", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.25)
    letter(ax, "A")

    ax = axes[1]
    order = ["rfstd", "nb", "iqr", "wstd", "std", "nnsim", "floor_g1.00"]
    names = {
        "rfstd": "model var.", "nb": "# nbrs", "iqr": "label IQR",
        "wstd": "dispersion", "std": "label std", "nnsim": "NN sim",
        "floor_g1.00": "γ=1 rule",
    }
    vals, cols, labs = [], [], []
    for tag in order:
        if tag not in P.index:
            continue
        vals.append(P.loc[tag, "d_cliff_mtd"] * 100)
        cols.append(HARM if tag == "floor_g1.00" else
                    (HARM if tag in ("rfstd", "nb") else
                     (DISP if tag in ("wstd", "std") else NEUTRAL)))
        labs.append(names[tag])
    xs = np.arange(len(vals))
    ax.bar(xs, vals, color=cols, edgecolor="white", zorder=3)
    ax.axhline(0, c=REF, lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=6.8, rotation=35, ha="right")
    ax.set_ylabel("cliff-group coverage vs global (pp)")
    ax.set_title("Matched width: no free lunch —\ngains vanish, harm survives", fontsize=9.5)
    for i, tag in enumerate([t for t in order if t in P.index]):
        p = P.loc[tag, "p_mtd"]
        st = sig(p)
        pos = P.loc[tag, "pos_mtd"]
        txt = f"{st}\n{pos}"
        if vals[i] >= 0:
            ax.text(i, vals[i] + 0.08, txt, ha="center", va="bottom", fontsize=6.8)
        else:
            ax.text(i, vals[i] - 0.08, txt, ha="center", va="top", fontsize=6.8)
    span = max(abs(min(vals)), abs(max(vals)))
    ax.set_ylim(-span * 1.35, span * 0.9)
    ax.grid(alpha=0.25, axis="y")
    letter(ax, "B")

    fig.suptitle(
        "Is conditioning the calibration set worth it?  "
        "15 ChEMBL assays × 10 splits, nominal 90% coverage", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/fig4_frontier.png")
    plt.close(fig)
    print("[saved] figures/fig4_frontier.png")


# ============================================================
# Fig 5 — 适用域对悬崖失效结构性失明（三面板）
# ============================================================
def fig5():
    S = pd.read_csv("results_ecfp_abl/AD_BY_SIMBIN.csv")
    T = pd.read_csv("results_ecfp_abl/AD_2x2.csv")
    C = pd.read_csv("results_ecfp_abl/AD_RECALL_CURVES.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.15]})

    ax = axes[0]
    x = S["sim_bin"].to_numpy(int)
    cov = S["coverage"].to_numpy(float) * 100
    cl = S["cliffB_rate"].to_numpy(float)
    ax.bar(x, cov, color=[NEUTRAL, NEUTRAL, BETTER, BETTER, BETTER],
           edgecolor="white", zorder=3, label="coverage of all molecules")
    for xi, ci in zip(x, cl):
        if ci > 0:
            ax.annotate(f"cliffs\n{ci:.1f}%", (xi, cov[xi]), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=8, color=HARM)
    ax.axhline(90, ls="--", c=REF, lw=1.1)
    ax.text(0.02, 91, "nominal 90%", fontsize=7.5, transform=ax.get_yaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(["0\n(sparsest)", "1", "2", "3", "4\n(densest)"], fontsize=8)
    ax.set_xlabel("nearest-neighbour similarity quintile")
    ax.set_ylabel("coverage / rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Denser = better covered,\nbut cliffs live only in bins 3–4", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.25, axis="y")
    letter(ax, "A")

    ax = axes[1]
    gname = T.columns[0]
    lab, val, col = [], [], []
    for g, cv in zip(T[gname].astype(str), T["coverage"].astype(float)):
        pretty = (g.replace("AD 内", "in-AD").replace("AD 外", "out-AD")
                  .replace("悬崖(cliffB)", "cliff").replace("非悬崖", "non-cliff"))
        lab.append(pretty)
        val.append(cv * 100)
        col.append(HARM if ("cliff" in pretty and "non-cliff" not in pretty) else NEUTRAL)
    xs = np.arange(len(val))
    ax.bar(xs, val, color=col, edgecolor="white", zorder=3)
    for xi, vi in zip(xs, val):
        ax.text(xi, vi + 1.0, f"{vi:.1f}", ha="center", fontsize=8.5)
    ax.axhline(90, ls="--", c=REF, lw=1.1)
    ax.set_xticks(xs)
    ax.set_xticklabels([l.replace(" × ", "\n× ") for l in lab], fontsize=8)
    ax.set_ylabel("coverage (%)")
    ax.set_ylim(0, 108)
    ax.set_title("All cliffs are inside the AD —\nand cost ~22 pp of coverage there", fontsize=9)
    ax.annotate("no cliff exists\noutside the AD", xy=(0.5, 0.10),
                xycoords="axes fraction", ha="center", fontsize=8,
                color=HARM, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=HARM, lw=0.8))
    ax.grid(alpha=0.25, axis="y")
    letter(ax, "B")

    ax = axes[2]
    f = 0.20
    C["s"] = C["rule"].astype(str).str.replace("（.*）", "", regex=True).str.strip()
    sub = C[np.isclose(C["flag_frac"], f)]
    EN = {"AD 规则": "AD rule\n(sparsest)",
          "离散度规则": "dispersion\n(most variable)",
          "合并": "combined"}
    names, vals, cols = [], [], []
    for s, v in zip(sub["s"], sub["failure_recall"]):
        names.append(EN.get(s, s))
        vals.append(v * 100)
        cols.append(DISP if "离散度" in s else (PURPLE if "合并" in s else NEUTRAL))
    xs = np.arange(len(vals))
    ax.bar(xs, vals, color=cols, edgecolor="white", zorder=3)
    ax.axhline(f * 100, ls="--", c=REF, lw=1.2)
    ax.text(0.98, 0.96, f"random flagging ({f:.0%})",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="top")
    for xi, vi in zip(xs, vals):
        ax.text(xi, vi + 0.9, f"{vi:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("failure recall (%) — all failures")
    ax.set_ylim(0, max(vals) * 1.42)
    ax.set_title("At a 20% review budget the two rules\nare complementary, not competing", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    letter(ax, "C")

    fig.suptitle(
        "Applicability-domain filtering is blind to activity-cliff failures "
        "— 70,170 molecules, 15 ChEMBL assays, 10 splits", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/fig5_ad_blindness.png")
    plt.close(fig)
    print("[saved] figures/fig5_ad_blindness.png")


# ============================================================
# Fig 6 — AD 结论的阈值/代理敏感性（两面板）
# ============================================================
def fig6():
    A = pd.read_csv("results_ecfp_abl/AD_SENSITIVITY.csv")
    A["short"] = A["proxy"].map(lambda s: "dist." if "距离" in s else "dens.")
    A["lab"] = A["short"] + "\n" + (A["ad_quantile"] * 100).round().astype(int).astype(str) + "%"

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    x = np.arange(len(A))
    w = 0.27

    ax = axes[0]
    ax.bar(x - w, A["fail_in_ad_cliff"] * 100, w, color=HARM, edgecolor="white",
           label="in-AD × activity cliff", zorder=3)
    ax.bar(x, (1 - A["cov_in_ad_noncliff"]) * 100, w, color=BETTER, edgecolor="white",
           label="in-AD × non-cliff", zorder=3)
    ax.bar(x + w, A["fail_rate_out_ad"] * 100, w, color=NEUTRAL, edgecolor="white",
           label="rejected by the AD", zorder=3)
    for xi, (a, b) in enumerate(zip(A["fail_in_ad_cliff"], A["cov_in_ad_noncliff"])):
        ax.annotate(f"+{(a - (1 - b)) * 100:.0f}pp", (xi - w, a * 100),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.5, color=HARM)
    ax.set_xticks(x)
    ax.set_xticklabels(A["lab"], fontsize=7.5)
    ax.set_xlabel("AD definition (type × quantile cut)")
    ax.set_ylabel("failure rate (%)")
    ax.set_ylim(0, 46)
    ax.set_title("Inside the AD, cliffs fail ~4× more\noften than non-cliffs — stable across all 8 cuts", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25, axis="y")
    letter(ax, "A")

    ax = axes[1]
    ax.bar(x, A["in_ad_recall"] * 100, color=HARM, edgecolor="white", zorder=3,
           label="in-AD failure recall (dispersion flag, 20% budget)")
    ax.axhline(20, ls="--", c=REF, lw=1.2)
    ax.text(0.02, 20.8, "random flagging (20%)", transform=ax.get_yaxis_transform(), fontsize=7.5)
    for xi, (r, l) in enumerate(zip(A["in_ad_recall"], A["in_ad_recall_lift"])):
        ax.text(xi, r * 100 + 0.6, f"{l:.2f}x", ha="center", fontsize=8, color=HARM)
    ax.set_xticks(x)
    ax.set_xticklabels(A["lab"], fontsize=7.5)
    ax.set_xlabel("AD definition (type × quantile cut)")
    ax.set_ylabel("failure recall (%)")
    ax.set_ylim(0, 52)
    ax.set_title("The label-free dispersion flag keeps\nthe same edge (1.78–1.95x) inside the AD", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    letter(ax, "B")

    fig.suptitle(
        "Robustness of the applicability-domain result "
        "(70,170 molecules, 15 ChEMBL assays, 10 splits)", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("figures/fig6_ad_sensitivity.png")
    plt.close(fig)
    print("[saved] figures/fig6_ad_sensitivity.png")


# ============================================================
# Fig 7 — 表示轴：ECFP4 vs RDKit 2D 描述符
# ============================================================
def fig7():
    a = pd.read_csv("results_ablation/PHASE1_ABLATION.csv")[["method", "delta_mean"]].rename(
        columns={"delta_mean": "ecfp"})
    b = pd.read_csv("results_xa_rdkit/PHASE1_ABLATION.csv")[["method", "delta_mean"]].rename(
        columns={"delta_mean": "alt"})
    d = a.merge(b, on="method", how="inner")
    d["short"] = d["method"].str.replace("mondrian_", "", regex=False)
    d = d.sort_values("ecfp", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [3, 1.4]})
    ax = axes[0]
    x = np.arange(len(d))
    w = 0.38
    ax.bar(x - w / 2, d["ecfp"], w, label="ECFP4 (2048 bits)", color=BETTER, alpha=0.9, zorder=3)
    ax.bar(x + w / 2, d["alt"], w, label="RDKit 2D descriptors", color=HARM, alpha=0.9, zorder=3)
    ax.axhline(0, ls="--", c=REF, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(d["short"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("cliff-group coverage change (pp)")
    ax.set_title("Same partition variable, two molecular representations", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    for i, (e, s) in enumerate(zip(d["ecfp"], d["alt"])):
        if np.sign(e) != np.sign(s):
            ax.annotate("sign flip", (i, min(e, s) - 0.004), ha="center",
                        fontsize=8, color=HARM)
    letter(ax, "A")

    ax = axes[1]
    try:
        m1 = pd.read_csv("results_phase2/C_misalignment.csv")
        m2 = pd.read_csv("results_xa_rdkit/C_misalignment.csv")
    except Exception as e:
        print("  [skip Fig7B] ", e)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig("figures/fig7_representation.png")
        plt.close(fig)
        return
    m = m1[["signal", "gap_predict_minus_see"]].merge(
        m2[["signal", "gap_predict_minus_see"]], on="signal", suffixes=("_ecfp", "_alt")
    ).sort_values("gap_predict_minus_see_ecfp", ascending=False)
    y = np.arange(len(m))
    ax.barh(y - w / 2, m["gap_predict_minus_see_ecfp"], w, color=BETTER, alpha=0.9, label="ECFP4")
    ax.barh(y + w / 2, m["gap_predict_minus_see_alt"], w, color=HARM, alpha=0.9,
            label="RDKit 2D")
    ax.axvline(0, ls="--", c=REF, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(m["signal"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("misalignment  M(s)")
    ax.set_title("Misalignment per signal", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, axis="x")
    letter(ax, "B")

    fig.tight_layout()
    fig.savefig("figures/fig7_representation.png")
    plt.close(fig)
    print("[saved] figures/fig7_representation.png")


# ============================================================
# Fig 8 — 阈值敏感性热图
# ============================================================
def fig8():
    from matplotlib.colors import TwoSlopeNorm
    A = pd.read_csv("results_sens/SENSITIVITY.csv")
    CFG_ORDER = ["base", "sim075", "sim095", "act050", "act150", "bins03", "bins08"]
    CFG_LABEL = {
        "base": "base", "sim075": "sim 0.75",
        "sim095": "sim 0.95", "act050": "act 0.5",
        "act150": "act 1.5", "bins03": "bins 3", "bins08": "bins 8",
    }
    VAR_ORDER = ["mondrian_nnsim", "mondrian_wstd", "mondrian_std", "mondrian_iqr",
                 "mondrian_pair", "mondrian_pairgap", "mondrian_nb", "mondrian_rfstd"]
    VAR_LABEL = {
        "mondrian_nnsim": "NN similarity", "mondrian_wstd": "label dispersion (wstd)",
        "mondrian_std": "label std", "mondrian_iqr": "label IQR",
        "mondrian_pair": "cliff pair present", "mondrian_pairgap": "max label gap in nbhd",
        "mondrian_nb": "# training neighbours", "mondrian_rfstd": "model-side variance",
    }
    UNUSABLE = {"sim095"}
    A = A[A["method"].isin(VAR_ORDER) & A["config"].isin(CFG_ORDER)]
    CFG_ORDER = [c for c in CFG_ORDER if c in set(A["config"])]

    piv = A.pivot_table(index="method", columns="config", values="delta_mean").reindex(
        index=VAR_ORDER, columns=CFG_ORDER)
    con = A.pivot_table(index="method", columns="config", values="n_ds_delta_pos").reindex(
        index=VAR_ORDER, columns=CFG_ORDER)
    nds = A.pivot_table(index="method", columns="config", values="n_ds").reindex(
        index=VAR_ORDER, columns=CFG_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0), gridspec_kw={"width_ratios": [1.55, 1.05]})

    ax = axes[0]
    M = piv.to_numpy(float)
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M * 100, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0, vmin=-vmax * 100, vmax=vmax * 100),
                   aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            cfg = CFG_ORDER[j]
            if cfg in UNUSABLE:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5, color="#666666")
                continue
            ax.text(j, i, f"{v*100:+.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.03 else "black")
    for j, cfg in enumerate(CFG_ORDER):
        if cfg in UNUSABLE:
            ax.axvspan(j - 0.5, j + 0.5, color="white", alpha=0.72, hatch="///", zorder=2)
    ax.set_xticks(range(len(CFG_ORDER)))
    ax.set_xticklabels([CFG_LABEL[c] for c in CFG_ORDER], fontsize=7.5,
                       rotation=30, ha="right")
    ax.set_yticks(range(len(VAR_ORDER)))
    ax.set_yticklabels([VAR_LABEL[v] for v in VAR_ORDER], fontsize=8.5)
    ax.set_title("Effect on cliff-group coverage (pp)\nblue = improves, red = harms", fontsize=9.5)
    fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02, label="Δ cliff coverage (pp)")
    letter(ax, "A")

    ax = axes[1]
    C = (con / nds.replace(0, np.nan)).to_numpy(float)
    Cm = np.ma.masked_invalid(C)
    im2 = ax.imshow(Cm, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            cfg = CFG_ORDER[j]
            if cfg in UNUSABLE or not np.isfinite(C[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5, color="#666666")
                continue
            n_pos, n_tot = int(con.to_numpy()[i, j]), int(nds.to_numpy()[i, j])
            ax.text(j, i, f"{n_pos}/{n_tot}", ha="center", va="center", fontsize=7.5, color="black")
    for j, cfg in enumerate(CFG_ORDER):
        if cfg in UNUSABLE:
            ax.axvspan(j - 0.5, j + 0.5, color="white", alpha=0.72, hatch="///", zorder=2)
    ax.set_xticks(range(len(CFG_ORDER)))
    ax.set_xticklabels([CFG_LABEL[c] for c in CFG_ORDER], fontsize=6.5, rotation=38, ha="right")
    ax.set_yticks(range(len(VAR_ORDER)))
    ax.set_yticklabels([VAR_LABEL[v] for v in VAR_ORDER], fontsize=8.5)
    ax.set_title("Direction consistency\n(# datasets positive / total)", fontsize=9.5)
    fig.colorbar(im2, ax=ax, shrink=0.82, pad=0.02, label="consistency")
    letter(ax, "B")

    fig.suptitle(
        "Do the conclusions survive threshold changes?  "
        "15 ChEMBL assays × 5 splits per config, nominal 90% coverage", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/fig8_threshold_sensitivity.png")
    plt.close(fig)
    print("[saved] figures/fig8_threshold_sensitivity.png")


def main():
    os.makedirs("figures", exist_ok=True)
    for name, fn in [("Fig2", fig2), ("Fig3", fig3), ("Fig4", fig4),
                     ("Fig5", fig5), ("Fig6", fig6), ("Fig7", fig7), ("Fig8", fig8)]:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
