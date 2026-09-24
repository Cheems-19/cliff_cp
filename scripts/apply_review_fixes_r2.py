# -*- coding: utf-8 -*-
"""Revision round 2 (user review items 2.1-2.5) — deterministic apply with assertions.

Targets:
  论文全文_v1.md      24 edits (abstract split, internal-note purge, 3.1-3.3 reorg,
                      systematic M x net_mag paragraph, AD per-definition ratios,
                      Tox21 group size, impl-disclosure -> SI pointer)
  SI_v1.md             1 edit  (prereg change-log sentence in S6.4)
  scripts/build_paper_html.py  1 edit (stale status banner)
All files are pure CRLF: normalize -> edit -> restore.
"""
import io
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\Users\Administrator\Desktop\小论文\cliff_cp"


def load(path):
    with io.open(f"{ROOT}/{path}", "r", encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


def save(path, s):
    with io.open(f"{ROOT}/{path}", "w", encoding="utf-8", newline="") as fh:
        fh.write(s.replace("\n", "\r\n"))


def rep(s, old, new, tag):
    n = s.count(old)
    assert n == 1, f"[{tag}] anchor count={n} (expect 1)\n--- old ---\n{old[:300]}"
    return s.replace(old, new)


# ============================================================ 论文全文_v1.md
s = load("论文全文_v1.md")
EDITS = []

EDITS.append(("title-meta",
"（初稿 v1 —— 合并自《引言与方法》《结果与讨论》两份草稿；图表为草稿版，定稿前需重绘）",
"（初稿 v3 · 2026-09-24）"))

EDITS.append(("abstract-split",
"而按预测类做类条件化毫无帮助；根源是**信号错配**：",
"""而按预测类做类条件化毫无帮助。

根源是**信号错配**："""))

EDITS.append(("abstract-ratio",
"域内悬崖分子的失败率是域内非悬崖分子的 **3.4–5.3 倍**（8 种界定全部成立）。因此",
"域内悬崖分子的失败率是域内非悬崖分子的 **3.6–5.3 倍**（2 族代理 × 4 阈值共 8 种界定逐项成立）。因此"))

EDITS.append(("del-1.5",
"""### 1.5 主张与证据一览（见骨架文档第八节）

十一条可写主张，每条附稳健性范围；两条被主动降级。
**写作纪律**：主结论用 8/8 稳健的那条（域内悬崖 vs 域内非悬崖），
"比被拒分子还差"这条阈值依赖（10/15–15/15），只作辅助并带范围。预注册原文见 SI §S7。

---
""",
"""---
"""))

EDITS.append(("s231-xref",
"（§3.2 验证其预测力；证明与经验验证见 SI §S5）",
"（§3.3 验证其预测力；证明与经验验证见 SI §S5）"))

EDITS.append(("discipline-note",
"都必须带范围引用**（骨架文档纪律）。",
"都必须带范围引用**。"))

EDITS.append(("holm-si",
"控制[44]（见预注册分析计划，原文见 SI §S7）。",
"控制[44]（预注册分析计划见 SI §S7）。"))

EDITS.append(("del-checklist",
"""---

## 附：定稿前核对清单（状态）

1. ✅ **相关工作的正式版**：见 §1.3（2026-09-23 复核：arXiv:2605.05562 的
   Mondrian 扩大组间差距与占位一致；"错配量 → 伤害"与"AD 对悬崖失明的量化"
   在化学域仍未见占位；补引 Norinder 2014 作 CP-as-AD 奠基引用）。
2. ✅ **图表**：图 1–12 已按 **600 dpi** 出版质量重绘，投稿图件包见
   `figures_submission/`（12 张 TIFF + 12 张矢量 PDF，双栏 17.4 cm）。
3. ✅ **术语统一**：`cliffB` 已改称"高活性悬崖分子"，
   首次出现处给出与 `cliffA` 的关系（§2.1）。
4. ✅ **限制节**：骨架第八节 14 条限制已并入正文 §4.3。

---

## 3 结果""",
"""---

## 3 结果"""))

EDITS.append(("sec31-header",
"### 3.1 共形失效集中在活性悬崖（图 1、图 2）",
"### 3.1 现象：共形失效集中在活性悬崖，且不随校准粒度消失（图 1、图 14）"))

EDITS.append(("sec31-scaffold2",
"这排除了\"把校准做局部就能修好\"的路线（骨架主张 2）。",
"这排除了\"把校准做局部就能修好\"的路线。"))

EDITS.append(("sec31-cut-auc",
"""本文因此对每个幅度结论都标注其划分协议。
模型侧不确定性对这一差距"视而不见"：随机森林树间方差在悬崖子集内的判别 AUC
只有 **0.468**（低于随机），在非悬崖子集内却有 0.625；
梯度提升集成方差为 0.487 / 0.625，D-MPNN 集成方差为 0.538 / 0.547
（D-MPNN 在悬崖内外都接近随机，见 §3.6）。

图 1 给出最直观的证据。""",
"""本文因此对每个幅度结论都标注其划分协议。

图 1 给出最直观的证据。"""))

EDITS.append(("sec32-merge",
"""![图2](fig2_mechanism.png)

**图2**　机制：左图显示只有标签离散度（d_wstd）能单调排出悬崖，模型方差（rf_std）几乎不能；右图显示 rf_std 在低分位箱收紧分位数，导致悬崖组覆盖率从 0.659 崩至 0.276，而 d_wstd 最高分位箱的悬崖组覆盖率从 0.685 升至 0.838。

### 3.2 机制：错配量预测条件化的伤害方向（图 10）

**错配量**定义为""",
"""### 3.2 机制：错配的分区信号把覆盖从悬崖组搬走（图 2、图 10）

机制的第一条证据是**组内判别力**（§2.3 的关键诊断）：模型侧不确定性在悬崖子集内的判别 AUC 接近或低于随机——随机森林树间方差 **0.468**、梯度提升集成方差 0.487、D-MPNN 集成方差 0.538——而在非悬崖子集内明显更高（0.625 / 0.625 / 0.547）：**它只在非悬崖上有信息量**（D-MPNN 在悬崖内外都接近随机，见 §3.6）。把这样的信号拿去做 Mondrian 分箱，后果如图 2 所示：`rf_std` 在低分位箱收紧分位数，悬崖组覆盖率从 0.659 崩至 0.276，而标签离散度 `d_wstd` 最高分位箱的悬崖组覆盖率从 0.685 升至 0.838。

![图2](fig2_mechanism.png)

**图2**　机制：左图显示只有标签离散度（d_wstd）能单调排出悬崖，模型方差（rf_std）几乎不能；右图显示 rf_std 在低分位箱收紧分位数，导致悬崖组覆盖率从 0.659 崩至 0.276，而 d_wstd 最高分位箱的悬崖组覆盖率从 0.685 升至 0.838。

**错配量**定义为"""))

EDITS.append(("sec32-systematic",
"""- **`d_nn_sim` 是明确的反例**：错配量最大（M=+0.40~0.47）却不伤害（−0.8 ~ +1.9 pp）。

**因此错配量应作为甄别规则使用，而不是定量预测公式**：""",
"""- **`d_nn_sim` 是明确的反例**：错配量最大（M=+0.40~0.47）却不伤害（−0.8 ~ +1.9 pp）。

把全部 8 个信号放在一起看，可以精确刻画这个反例的边界（M 取四族范围；net_mag 与实测 ΔCov 为 RF 族、两个表示的均值，即图 12 的数据）：`d_pair`/`d_pair_gap` 的 M = −0.19~−0.18、net_mag ≈ 0（−0.004）、实测 −0.3/−0.6 pp；`d_iqr` 的 M = −0.04~−0.02、net_mag +0.033、实测 +0.5 pp；`d_wstd`/`d_std` 的 M ≈ −0.03~0.00、net_mag +0.063/+0.060、实测 +1.7/+1.9 pp；`d_nb` 的 M = +0.14~+0.16、net_mag −0.046、实测 −3.6 pp；`rf_std` 的 M = +0.05~+0.18（族依赖：D-MPNN 族仅 +0.054，该族伤害也接近消失，+0.4 pp）、net_mag 在两表示间为 −0.033~+0.001、实测 −5.2/−6.5 pp；`d_nn_sim` 的 M = +0.40~+0.47、net_mag +0.037、实测 +1.7 pp。据此，**落入「M>0 但净加宽」区域的信号只有一个：`d_nn_sim`**；M>0 的另外两个信号（`d_nb`、`rf_std`）都落在净收紧（或中性偏收紧）方向且显著伤害。`d_nn_sim` 为何落入该区已有直接检验：**分箱约定的产物**——无邻居分子经 `np.digitize` 全部落入顶箱、抬高顶箱分位数，恰好保护了同箱的悬崖分子（若按 NaN→0 落箱，同一信号约 −12 pp），见 §4.3 条 13。

**因此错配量应作为甄别规则使用，而不是定量预测公式**："""))

EDITS.append(("sec33-merge",
"""**可部署诊断的验证（图 12）**：命题 1 的恒等式在 240 个 (数据集 × 信号) 组上逐分子
精确成立（最大残差 1.0×10⁻¹⁶）。式 (5) 的 `net_mag` 与观测到的悬崖组覆盖变化，
在两个分子表示上秩相关 **Spearman ρ = 0.865**（n=16 = 8 信号 × 2 表示）：
净收紧的信号（`d_nb`、`rf_std`）预测为负且实测为负（−3.1 ~ −6.5 pp），
净加宽的信号（`d_wstd`/`d_std`/`d_nn_sim`）预测为正且实测为正（+1.6 ~ +2.1 pp），
而错配量≈0 且无判别力的 `d_iqr` 预测出小幅加宽、实测接近 0（+0.3 ~ +0.8 pp）。
该相关关系的统计强度：**置换检验 p = 2×10⁻⁵**（B = 2×10⁵，打乱对应关系），
bootstrap 95% 置信区间 **[+0.64, +0.94]**，**留一法**下 ρ 仅在 [+0.84, +0.89] 间变动
（100% 的剔除后仍 > 0.7，无单点杠杆）。若按**信号**聚合（每个信号在两个表示上取均值，
n=8，消除表示轴的相关性），ρ = **+0.929**，置换 p = 0.0027。
**因此本文不仅解释"为什么"（错配量），还给出一个部署前可算、不看测试标签的判据**——
比事后 AUC 更接近可操作的建议。

![图12](fig12_transfer_diagnostic.png)

**图12**　覆盖转移的可部署诊断：横轴 net_mag（式 5，仅用校准集与训练集信息，无测试标签），纵轴观测到的悬崖组覆盖变化。Spearman ρ = 0.865（n=16）。净收紧的 d_nb / rf_std 落在左下（预测受害且实测受害），净加宽的 d_wstd / d_std / d_nn_sim 落在右上，无判别力的 d_iqr 贴近零线。

### 3.3 同等覆盖率下：只有对齐信号给出"选择性"改善（表 2、图 4）

![图3](fig3_ablation.png)

**图3**　分区变量消融：收益（悬崖组覆盖变化）vs 代价（区间宽度变化）。标签离散度族位于左上（小代价、正收益），模型方差/密度/相似度侧位于右下（更窄但伤害悬崖组）。

朴素比较""",
"""### 3.3 可操作诊断：部署前判据与同等覆盖率下的安全信号（图 12、表 2、图 4）

**可部署诊断的验证（图 12）**：命题 1 的恒等式在 240 个 (数据集 × 信号) 组上逐分子
精确成立（最大残差 1.0×10⁻¹⁶）。式 (5) 的 `net_mag` 与观测到的悬崖组覆盖变化，
在两个分子表示上秩相关 **Spearman ρ = 0.865**（n=16 = 8 信号 × 2 表示）：
净收紧的信号（`d_nb`、`rf_std`）预测为负且实测为负（−3.1 ~ −6.5 pp），
净加宽的信号（`d_wstd`/`d_std`/`d_nn_sim`）预测为正且实测为正（+1.6 ~ +2.1 pp），
而错配量≈0 且无判别力的 `d_iqr` 预测出小幅加宽、实测接近 0（+0.3 ~ +0.8 pp）。
该相关关系的统计强度：**置换检验 p = 2×10⁻⁵**（B = 2×10⁵，打乱对应关系），
bootstrap 95% 置信区间 **[+0.64, +0.94]**，**留一法**下 ρ 仅在 [+0.84, +0.89] 间变动
（100% 的剔除后仍 > 0.7，无单点杠杆）。若按**信号**聚合（每个信号在两个表示上取均值，
n=8，消除表示轴的相关性），ρ = **+0.929**，置换 p = 0.0027。
**因此本文不仅解释"为什么"（错配量），还给出一个部署前可算、不看测试标签的判据**——
比事后 AUC 更接近可操作的建议。

![图12](fig12_transfer_diagnostic.png)

**图12**　覆盖转移的可部署诊断：横轴 net_mag（式 5，仅用校准集与训练集信息，无测试标签），纵轴观测到的悬崖组覆盖变化。Spearman ρ = 0.865（n=16）。净收紧的 d_nb / rf_std 落在左下（预测受害且实测受害），净加宽的 d_wstd / d_std / d_nn_sim 落在右上，无判别力的 d_iqr 贴近零线。

朴素比较"""))

EDITS.append(("fig3-before-fig4",
"""代价由悬崖组承担。

![图4](fig4_frontier.png)""",
"""代价由悬崖组承担。

![图3](fig3_ablation.png)

**图3**　分区变量消融：收益（悬崖组覆盖变化）vs 代价（区间宽度变化）。标签离散度族位于左上（小代价、正收益），模型方差/密度/相似度侧位于右下（更窄但伤害悬崖组）。

![图4](fig4_frontier.png)"""))

EDITS.append(("sec35-ratios",
"""- **AD 内悬崖分子的失败率（26–28%）是同域非悬崖分子（5–8%）的 3.4–5.3 倍**——
  这一条在 2 族 AD 代理（距离型/密度型）× 4 个阈值共 8 种界定下**全部成立**；""",
"""- **AD 内悬崖分子的失败率（26–28%）是同域非悬崖分子（5–8%）的 3.6–5.3 倍**——
  逐界定配对比值：距离型 3.7 / 4.2 / 4.8 / 5.3，密度型 3.6 / 3.9 / 4.1 / 4.2
  （阈值为 10/20/30/40% 分位）——这一条在 2 族 AD 代理（距离型/密度型）× 4 个阈值
  共 8 种界定下**全部成立**；即使做最不利的跨界定配对（最低悬崖失败率 26.1% 对
  最高非悬崖失败率 7.8%），比值仍为 3.4；"""))

EDITS.append(("sec35-maindef",
"（域内悬崖 28.5% vs 域内非悬崖 6.7%，**4.2 倍**）——这一差异无法由相似度解释；",
"（域内悬崖 28.5% vs 域内非悬崖 6.7%，**4.2 倍**，距离型 20% 分位界定）——这一差异无法由相似度解释；"))

EDITS.append(("sec36-header",
"### 3.6 稳健性（图 7、8；表见骨架第八节）",
"### 3.6 稳健性（图 7、8）"))

EDITS.append(("rfstd-note",
"""  把成员从 3 增到 8 时所有分区变量的效应整体上移约 0.5 pp
  （该操作同时改变了模型，故不能单独检验 rf_std 通道，正文如实说明）。""",
"""  把成员从 3 增到 8 时所有分区变量的效应整体上移约 0.5 pp；该操作同时改变了
  模型与信号，无法单独检验 rf_std 通道（§4.3 条 10）。"""))

EDITS.append(("impl-disclosure",
"**实现说明（如实披露）**：表 4 中 Mondrian ← `d_wstd` 与 Mondrian ← `d_nb` 两行首次运行时，`d_wstd` 对**无训练邻居分子**留下的 NaN 使分位数边界全部退化为 NaN，所有分子落入同一箱；该箱样本数必然达标，于是回退成全局分位数，整臂**静默退化成 global**（表现为 Δ 恰为 0.00 pp、方向一致 0/15、Wilcoxon p = nan）。该问题已定位，并按主流水线既定约定修复（NaN 归入最稀疏箱）后补跑，表 4 两行即补跑数字。**补跑与首跑的可复现性已交叉校验**：两跑在同一 (划分, 数据集, 种子) 上给出的 `mondrian_rfstd` 覆盖率**逐点完全一致（150 组，最大绝对差 0.000）**，故两行可安全拼接进同一张表。修订已记入预注册文档的\"本计划之外的改动记录\"。",
"表 4 中 Mondrian ← `d_wstd` 与 Mondrian ← `d_nb` 两行涉及一次已定位并修复的实现缺陷：`d_wstd` 对无训练邻居分子留下的 NaN 曾使分位数边界退化、整臂静默退化成 global（两行首跑表现为 Δ = 0.00 pp、方向一致 0/15）。缺陷机理见 SI §S1.3；修复后按主流水线约定补跑，补跑与首跑在同一 (划分, 数据集, 种子) 上的 `mondrian_rfstd` 覆盖率逐点完全一致（150 组，最大绝对差 0.000），复现性交叉校验见 SI §S6.4。"))

EDITS.append(("report-discipline",
"> **报告纪律**（不变）：",
"> **报告纪律**："))

EDITS.append(("sec310-cliffcov",
"""  HIV 与 Tox21-NR-AR 的标签悬崖覆盖近乎 0（0.040 / 0.027）——邻居标签冲突的分子
  本质上就是模型学不到的分子，与回归里 cliffB 的机制同源。""",
"""  HIV 与 Tox21-NR-AR 的标签悬崖覆盖接近 0（0.040 / 0.027）。以悬崖群规模计
  （HIV 平均 36.6 个/种子，Tox21-NR-AR 平均仅 7.2 个、范围 4–10），Tox21-NR-AR 的
  0.027 意味着平均每个种子只有约 0.2 个悬崖分子被覆盖——该读数应视为**小样本下的
  描述性结果**而非精确估计；但其方向在 10/10 种子上一致，且与回归里 cliffB 的机制同源：
  邻居标签冲突的分子本质上就是模型学不到的分子。"""))

EDITS.append(("sec310-note",
"""（注：标签悬崖群较小（每种子平均 BBBP 2.4 个、ClinTox 1.5 个，ClinTox 部分种子无悬崖分子），故 BBBP 7/10、ClinTox 5/10 的
方向一致主要体现趋势而非高功效结论；BACE/HIV/Tox21-NR-AR 在 10/10 种子上一致，功效充足。
脚本 `scripts/phase17_classify.py`，结果 `results/phase17_classify_v2/`；
数据 MoleculeNet CC-BY，下载缓存于服务器 `data/classification/`。）""",
"""（注：标签悬崖群规模（每种子平均）：BACE 22 个、BBBP 2.4 个、ClinTox 1.5 个
（部分种子为 0）、HIV 36.6 个、Tox21-NR-AR 7.2 个（范围 4–10）。
故 BBBP 7/10、ClinTox 5/10 的方向一致主要体现趋势而非高功效结论；
BACE/HIV/Tox21-NR-AR 在 10/10 种子上一致。
脚本 `scripts/phase17_classify.py`，结果 `results/phase17_classify_v2/`；
数据 MoleculeNet CC-BY，下载缓存于服务器 `data/classification/`。）"""))

EDITS.append(("sec43-13",
"（`net_mag` = +0.037，实测 ΔCov = +1.7 pp，两者同号，见 §3.2 与图 12）。",
"""（`net_mag` = +0.037，实测 ΔCov = +1.7 pp，两者同号；它是 §3.2 逐信号清单中唯一落入
「M>0 但净加宽」区的信号，见 §3.3 与图 12）。"""))

for tag, old, new in EDITS:
    s = rep(s, old, new, tag)
save("论文全文_v1.md", s)
print(f"[ok] 论文全文_v1.md: {len(EDITS)} edits applied")

# ============================================================ SI_v1.md
t = load("SI_v1.md")
t = rep(t,
"so the re-run numbers can be spliced into the same table as the first-run numbers.",
"so the re-run numbers can be spliced into the same table as the first-run numbers. "
"The revision is recorded in the pre-registration document's change log (modifications "
"beyond the original plan).",
"si-s64-prereg")
save("SI_v1.md", t)
print("[ok] SI_v1.md: 1 edit applied")

# ============================================================ build_paper_html.py
b = load("scripts/build_paper_html.py")
b = rep(b,
'<div class="meta">初稿 v3 · 2026-09-23 · 图 1–12 已按 600 dpi 出版质量重绘 · §3.8 结果待填（phase11/13 运行中）</div>',
'<div class="meta">初稿 v3 · 2026-09-24 · 图 1–15 已按 600 dpi 出版质量重绘 · 表 4 / 表 5 已回填（§3.8）</div>',
"banner")
save("scripts/build_paper_html.py", b)
print("[ok] scripts/build_paper_html.py: 1 edit applied")

# ============================================================ read-back verification
v = load("论文全文_v1.md")
STALE = ["待填", "骨架文档", "骨架主张", "预注册原文见", "如实说明）", "如实披露",
         "fig612", "附：定稿前核对清单", "主张与证据一览", "3.4–5.3 倍", "近乎 0"]
NEED = ["3.6–5.3 倍", "唯一落入「M>0 但净加宽」", "SI §S6.4", "现象：共形失效集中在活性悬崖",
        "可操作诊断：部署前判据", "错配的分区信号把覆盖从悬崖组搬走", "小样本下的",
        "Tox21-NR-AR 7.2 个（范围 4–10）", "2026-09-24"]
bad = [k for k in STALE if k in v]
missing = [k for k in NEED if k not in v]
assert not bad, f"stale markers remain: {bad}"
assert not missing, f"expected markers missing: {missing}"
n_abs = v.count("3.6–5.3 倍")
print(f"[verify] stale markers: 0; expected markers all present; '3.6–5.3 倍' x{n_abs} (expect 2)")
si = load("SI_v1.md")
assert "change log" in si
print("[verify] SI_v1.md: change-log sentence present")
print("ALL DONE")
