# -*- coding: utf-8 -*-
"""修订轮 4：三份修改建议（第二轮）落地。
内容：摘要 AD 论证重写/六级谱、图号漏改、残句删除、rf_std 四项、
AD 论证重构、[50] 划界、表2/3 口径注、E2 删逐行 p、预注册措辞、
R1—→R1：、初步发现→发现、§2.9 扩充、run 单位统一、Venn/Mondrian 补引、
MoleculeACE 许可、[50] 作者全名、全文引用重编号（按首次出现顺序）。
"""
import io
import re

ROOT = "."
TAGS = []


def load(path):
    with io.open(f"{ROOT}/{path}", "r", encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


def save(path, s):
    with io.open(f"{ROOT}/{path}", "w", encoding="utf-8", newline="") as fh:
        fh.write(s.replace("\n", "\r\n"))


def rep(s, old, new, tag):
    n = s.count(old)
    assert n == 1, f"[{tag}] anchor count={n}: {old[:60]!r}"
    TAGS.append(tag)
    return s.replace(old, new)


# ============ 第一部分：论文正文内容编辑 ============
p = "论文全文_v1.md"
s = load(p)

# e01 标题版本号
s = rep(s, "（初稿 v4 · 2026-09-24）", "（初稿 v5 · 2026-09-24）", "e01-title")

# e02 摘要：六级谱表述
s = rep(s,
    "把两个划分点扩展为六级偏移强度谱后：",
    "把划分协议扩展为按强度递增的六级偏移强度谱（从随机划分到向更高效力的活性外推）后：",
    "e02-abstract-spectrum")

# e03 摘要：AD 论证重写（跨行）
old_ad = ("任何基于\"离训练集远即不可靠\"的机制——包括适用域——都对此结构性失明：40.8%–75.9% 的失效发生在适用域内部，\n"
          "而悬崖只出现在相似度最高的两层、即 AD 判为最可信的位置，域内悬崖分子的失败率是域内非悬崖分子的 **3.6–5.3 倍**（2 族代理 × 4 阈值共 8 种界定逐项成立）。")
new_ad = ("任何基于\"离训练集远即不可靠\"的机制——包括适用域（AD）——都够不着悬崖失效，但 AD 的角色必须说清：它对失效本身有真实的域外预警力"
          "（域外分子占测试集 10.2–40.0%，却承载 24.1–59.2% 的失效，富集 1.5–2.4 倍），其预警力却全部来自距离/密度轴、与悬崖轴**正交**"
          "——悬崖只出现在相似度最高、AD 判为最可信的位置，且在同一 AD 界定内部，悬崖分子的失败率是同域非悬崖分子的 **3.6–5.3 倍**（8 种界定逐项成立）。")
s = rep(s, old_ad, new_ad, "e03-abstract-AD")

# e04 摘要：删章节号
s = rep(s, "（覆盖率匹配口径下，CQR 之后仍残余 8.6 pp 差距，§3.4）。",
        "（覆盖率匹配口径下，CQR 之后仍残余 8.6 pp 差距）。", "e04-abstract-sec")

# e05 §1.4 与摘要一致（×7）
s = rep(s, "（3 模型族 × 2 表示 × 7 阈值配置），并在**两种划分协议**",
        "（3 模型族 × 2 表示；阈值敏感性另做 7 配置单因子扫描，见 §3.3），并在**两种划分协议**",
        "e05-s14-thresholds")

# e06 §1.4 AD 贡献措辞
old_c3 = "3. **与 AD 的关系**：量化 AD 对悬崖失效的结构性失明，并给出一个\n   不含查询标签的域内预警信号。"
new_c3 = ("3. **与 AD 的关系**：量化 AD 对悬崖失效的\"正交失明\"——AD 有真实的域外预警力（失效在域外富集 1.5–2.4 倍），"
          "但其预警力与悬崖轴正交——并给出一个不含查询标签的域内预警信号。")
s = rep(s, old_c3, new_c3, "e06-s14-AD")

# e07 §2.2 Mondrian 段补方法学源头引用 [1]（Vovk 2005 专著）
s = rep(s, "**Mondrian 条件化**：按某个**无标签信号** `d(·)` 把校准集与测试集分为 K 组",
        "**Mondrian 条件化**[1]：按某个**无标签信号** `d(·)` 把校准集与测试集分为 K 组",
        "e07-mondrian-cite")

# e08 §2.2 Venn–ABERS 补讨论
s = rep(s, "`w(x) = p_test(x)/p_cal(x)`，对校准分数取加权分位数。实现参照 MAPIE[12]。",
        "`w(x) = p_test(x)/p_cal(x)`，对校准分数取加权分位数。实现参照 MAPIE[12]。"
        "另一族分布无关方法是 Venn 预测器（含 Venn–ABERS 变体）[1]，不依赖 α 且直接输出校准概率集，"
        "但不提供\"按信号分组\"的诊断维度，与本文的机制问题正交，故不展开。",
        "e08-venn")

# e09 §2.2 4 个「模型×表示」组合说明
s = rep(s, "D-MPNN[35]（chemprop 2.3.1[36]，`d_h=300`，`depth=3`，30 epochs，patience 5，3 成员集成）。",
        "D-MPNN[35]（chemprop 2.3.1[36]，`d_h=300`，`depth=3`，30 epochs，patience 5，3 成员集成）。\n\n"
        "上述模型族与表示组合成 4 个「模型×表示」组合用于机制扫描（§3.2）：RF+ECFP4、GBR+ECFP4、RF+RDKit 描述符、"
        "D-MPNN（图输入；其邻域/离散度信号仍基于 ECFP4 邻域）。覆盖率匹配主分析（§3.3）固定在 RF 族的两种表示上。",
        "e09-four-combos")

# e10 表1 d_pair 命名统一（与数据文件一致）
s = rep(s, "| `d_pair_gap`（另实现记 `d_pairgap`） | 邻域内最大标签差（同一量的两种实现） | 数据侧 |",
        "| `d_pair` / `d_pair_gap` | 邻域内最大标签差（同一量的两种实现） | 数据侧 |",
        "e10-table1-dpair")

# e11 §2.6 限定主分析口径
s = rep(s, "结果：所有方法的边际覆盖都在名义值 ±0.3 pp 内（0.899–0.903）；",
        "结果：**主分析（随机划分、α=0.10）**中所有方法的边际覆盖都在名义值 ±0.3 pp 内（0.899–0.903）；"
        "骨架划分与偏移谱下的边际覆盖见 §3.8（骨架下 4/15 数据集低于 0.88）与 §3.9（活性偏移下整体击穿名义值）。",
        "e11-s26-scope")

# e12 §2.9 18.2 行补 10 种子
s = rep(s, "| 18.2 pp | 随机划分 · cliffB 机制定义 · α=0.10 · 原始覆盖率 | §3.1 |",
        "| 18.2 pp | 随机划分 · cliffB 机制定义 · α=0.10 · 原始覆盖率 · 10 种子 | §3.1 |",
        "e12-s29-182")

# e13 §2.9 扩充口径行
new_rows = ("| 18.1 pp（全量合并） | 预检阶段 8 数据集 · cliffB · 随机效应合并（CI 13.8–22.3） | §3.1 图 2B |\n"
            "| 18.1 pp（分层） | 最疏相似度十分位的 cliffB 差距（分层口径，与上一行不同源） | §3.9 发现 4、图 14C |\n"
            "| +13.2 → +30.1 pp（cliffB）/ +7.3 → +17.9 pp（cliffA） | α 扫描 α∈{0.07→0.20} · 4 种子 | §3.1 |\n"
            "| +19.8 → +15.8 → +10.3 pp（cliffB）/ +11.0 → +7.6 → +4.4 pp（cliffA） | 偏移谱结构轴 random→精确骨架→泛化骨架 · 3 种子 | §3.9 |\n"
            "| 0.188 / 0.284 | 活性偏移 40%/20% 下的**全体**边际覆盖（非悬崖组） | §3.9 发现 2 |\n"
            "| +10.4 / +16.0 pp | 自适应共形在活性偏移下的悬崖组覆盖改善 | §3.9 发现 3 |\n"
            "| 35.3% vs 13.3% | 下游低估风险 τ=0.75 · global · 随机划分 | §3.7 |\n"
            "| +35.7 ~ +88.0 pp | 分类标签悬崖 vs 非悬崖覆盖差距 · 5 基准 · 集合共形 | §3.10 |\n"
            "| +0.5 ~ +2.4 pp | `d_wstd`/`d_std` 逐组合实测 ΔCov · 原始覆盖率 · 机制扫描 | §3.2 |\n"
            "| +6.37 / +1.05 / −4.38 → −12.63 pp | 表 4 判读中的 cliffB 机制定义读数（随机/骨架） | §3.8 |")
s = rep(s, "| 4.2 倍 | 主界定（距离型 20% 分位）的同一比值 | §3.5 |",
        "| 4.2 倍 | 主界定（距离型 20% 分位）的同一比值 | §3.5 |\n" + new_rows,
        "e13-s29-rows")

# e14 §3.1 图 1 正文引用
s = rep(s, "本文因此对每个幅度结论都标注其划分协议。",
        "本文因此对每个幅度结论都标注其划分协议。\n\n"
        "图 1 给出这一失效的最小实例：一个 ECFP4 指纹完全相同的悬崖对，模型对两个分子给出同一预测与同一区间，"
        "却只有高效力一侧失效——覆盖差距在分子尺度上的具象。",
        "e14-fig1-ref")

# e15 图2 caption 补预检说明
s = rep(s, "三种校准粒度下差距均存在——局部化校准不能解决问题。",
        "三种校准粒度下差距均存在——局部化校准不能解决问题。注：B 为预检阶段 8 个最大数据集的合并（非全 15 集）；"
        "全量 10 种子口径的合并值为 18.2 pp（§3.1）。",
        "e15-fig2b-note")

# e16 §3.2 删残句
s = rep(s, "按 §2.3 的定义（`M(s) = AUC_fail(s) − AUC_cliff(s)`；`M>0` 表示\"看得见失效却看不见悬崖\"）。\n\n",
        "", "e16-s32-dead")

# e17 §3.2 run 单位统一
s = rep(s, "按 7 个独立信号（`d_pair` 与 `d_pair_gap` 为同一量的两种实现，故 8 个实现 × 4 族共 32 点、去重后 28 点；图 4）逐族扫描：",
        "按 7 个独立信号（`d_pair` 与 `d_pair_gap` 为同一量的两种实现）× 4 个「模型×表示」组合逐族扫描（图 4）。"
        "统计单位约定：1 run =（数据集, 种子, 信号实现, 表示）；本节 32 点 = 8 信号实现 × 4 组合"
        "（每点为 15 数据集 × 种子上的聚合），去重后 28 点；§3.3 的 n=16 = 8 信号实现 × 2 表示（RF 族），240 = 15 数据集 × 16 run。",
        "e17-run-unit")

# e18 §3.2 n=24 说明
s = rep(s, "Pearson r = **−0.63**（n=24）。",
        "Pearson r = **−0.63**（n=24 = 28 点剔除 4 个 `d_nn_sim` 点）。", "e18-n24")

# e19 §3.2 表头单位
s = rep(s, "| 信号 | M（四族范围） | net_mag (pp) | 实测 ΔCov (pp) | 判读 |",
        "| 信号 | M（四族范围） | net_mag（无量纲比值） | 实测 ΔCov (pp，原始覆盖率) | 判读 |",
        "e19-table-header")

# e20 §3.2 rf_std 脚注重写（口径 + 方向不一致）
old_fn = ("\\* `rf_std` 的实现级 net_mag（两表示分别 +0.001 / −0.033）与 SI §S5.5 合并判据的信号级 net_mag（−1.63，两表示平均）并不矛盾："
          "前者是 RF 族逐实现的读数，后者把两表示合并；该信号的真实伤害来自\"大幅收紧而校准侧质量小\"，"
          "幅度被实现级 net_mag 低估，故 S5.5 以箱分位数比值（q_ratio）补足（§4.3 条 13、SI §S5.5）。")
new_fn = ("\\* `rf_std` 的实现级 net_mag（两表示分别 +0.001 / −0.033，无量纲比值）与 SI §S5.5 合并判据的信号级 net_mag"
          "（−1.63 pp，两表示合并后的净收紧幅度，单位 pp）是**两个不同口径的量**，不可直接比较：前者是 RF 族逐实现的相对加宽幅度，"
          "后者是两表示合并后的信号级读数。且 `rf_std` 的一个表示 net_mag = +0.001（预测净加宽）而实测 ΔCov = −5.2 pp，"
          "属**方向不一致**而非\"幅度低估\"——该信号的真实伤害来自\"大幅收紧而校准侧质量小\"，方向判定由 S5.5 的箱分位数比值（q_ratio）补足"
          "（§4.3 条 13、SI §S5.5）。")
s = rep(s, old_fn, new_fn, "e20-rfstd-footnote")

# e21 §3.3 图5段：方向一致性如实表述
s = rep(s, "净收紧的信号（`d_nb`、`rf_std`）预测为负且实测为负（−3.1 ~ −6.5 pp），",
        "净收紧的信号实测为负（`d_nb`、`rf_std`，−3.1 ~ −6.5 pp；16 个 run 中 15 个预测方向与实测一致，"
        "唯一例外为 `rf_std` 的一个表示：net_mag = +0.001 而实测 −5.2 pp，见 §3.2 注），",
        "e21-s33-direction")

# e22 §3.3 口径标注
s = rep(s, "预测为正且实测为正（+1.6 ~ +2.1 pp），",
        "预测为正且实测为正（+1.6 ~ +2.1 pp，原始覆盖率口径），", "e22-s33-caliber")

# e23 表3 标题补口径注
s = rep(s, "**表 3**  方法阶梯：各方法在同等边际覆盖 0.90 下的悬崖组覆盖与宽度代价。",
        "**表 3**  方法阶梯：各方法在同等边际覆盖 0.90 下的悬崖组覆盖与宽度代价"
        "（覆盖率匹配批跑：每方法按各自 target α 网格插值，与表 2 的逐配对插值批跑不同源，两表 Δ 不可直接互推）。",
        "e23-table3-title")

# e24 表3 底层模型注
s = rep(s, "| Mondrian ← d_nb | 0.6713 | **23.8 pp** | +1.0% |\n\n四点结论：",
        "| Mondrian ← d_nb | 0.6713 | **23.8 pp** | +1.0% |\n\n"
        "注：global 与 Mondrian 各行的底层模型为随机森林（同模型对照）；CQR 以梯度提升（HGB）为底层模型（§2.2），"
        "其改善包含底层模型差异的贡献，不可全部归因于共形化本身。\n\n四点结论：",
        "e24-table3-models")

# e25 §3.5 标题图号
s = rep(s, "### 3.5 适用域对悬崖失效结构性失明（图 9、6）",
        "### 3.5 适用域对悬崖失效结构性失明（图 9、图 10）", "e25-s35-title")

# e26 §3.5 第一条 bullet 重写（AD 富集论证）
s = rep(s, "- 共形失效发生在 AD **内部**的比例为 40.8%–75.9%（视 AD 界定宽严）；",
        "- 失效的域内外分布：AD **内部**承载 40.8%–75.9% 的失效（视界定宽严），而域内分子占测试集 60.0%–89.8%——"
        "即**域外**以 10.2%–40.0% 的分子承载 24.1%–59.2% 的失效，**富集 1.5–2.4 倍**："
        "AD 作为\"不可靠\"过滤器有真实的（弱）预警力，与下文图 9C 的召回一致（AD 规则 38.2% > 随机 20%）；",
        "e26-s35-enrich")

# e27 §3.5 第二条 bullet 正交化
s = rep(s, "- **AD 内悬崖分子的失败率（26–28%）是同域非悬崖分子（5–8%）的 3.6–5.3 倍**——",
        "- **但其预警力与悬崖轴正交：AD 内悬崖分子的失败率（26–28%）是同域非悬崖分子（5–8%）的 3.6–5.3 倍**——",
        "e27-s35-orthogonal")

# e28 §3.6 标题图号
s = rep(s, "### 3.6 稳健性（图 11、8）", "### 3.6 稳健性（图 11、图 12）", "e28-s36-title")

# e29 §3.8 R1/R2/R3 措辞
s = rep(s, "**R1 —— 现象在骨架划分下仍然成立（S6 通过）。**",
        "**R1：现象在骨架划分下仍然成立（S6 通过）。**", "e29-r1")
s = rep(s, "**R2 —— 补充基线不消除差距（S7 通过），但自适应共形是本组最强非 CQR 方法。表 4 给出全部数字。**",
        "**R2：补充基线不消除差距（S7 通过），但自适应共形是本组最强非 CQR 方法。表 4 给出全部数字。**", "e29-r2")
s = rep(s, "**R3 —— 固定预测器下，信号精度与伤害单调相关（受控机制检验通过）。**",
        "**R3：固定预测器下，信号精度与伤害单调相关（受控机制检验通过）。**", "e29-r3")

# e30 §3.8 预注册措辞
s = rep(s, "本节对应预注册的 **S6–S7**（**S8 决策影响已在 §3.7 报告**）；判读标准写于数据之前，逐条对照见本小节末尾。",
        "本节对应预注册分析计划的 **S6–S7**（判读标准在数据分析前写定于内部预注册文档 SI §S7，"
        "未在 OSF/AsPredicted 公开注册，暂无注册号；**S8 决策影响已在 §3.7 报告**），逐条对照见本小节末尾。",
        "e30-prereg")

# e31 表4 cliffA 标注
s = rep(s, "**表 4　覆盖差距与宽度代价：随机 vs 骨架划分 × 8 方法**（α = 0.10，15 数据集 × 5 种子；Δ = 相对 global 的悬崖组覆盖率变化；宽度为相对 global 的百分比变化）",
        "**表 4　覆盖差距与宽度代价：随机 vs 骨架划分 × 8 方法**（α = 0.10，15 数据集 × 5 种子；"
        "Δ = 相对 global 的悬崖组覆盖率变化，主列为数据集级 cliffA，机制定义 cliffB 的读数在判读中标注；宽度为相对 global 的百分比变化）",
        "e31-table4")

# e32 表5 列名解释
s = rep(s, "**表 5　信号精度阶梯（固定预测器，15 数据集 × 5 种子；Δ 为悬崖组覆盖率变化）**",
        "**表 5　信号精度阶梯（固定预测器，15 数据集 × 5 种子；Δ 为悬崖组覆盖率变化，A 列 = cliffA 数据集级口径、B 列 = cliffB 机制定义口径）**",
        "e32-table5")

# e33 E2 删逐行 p
s = rep(s, "（0.592 → 0.722；按 k 聚合 Spearman ρ = **+1.000**，逐行 +0.618，p = 1.1×10⁻⁵⁶；n = 525 行 = 15 数据集 × 5 种子 × 7 档信号精度）",
        "（0.592 → 0.722；按 k 聚合 Spearman ρ = **+1.000**；n = 525 行 = 15 数据集 × 5 种子 × 7 档信号精度，"
        "行间共享模型与数据集、彼此不独立，故不报逐行检验，推断以聚合 ρ 与下条剂量效应检验为准）",
        "e33-e2-p")

# e34 判读表下临界通过注
s = rep(s, "\n至此 **S6–S8 全部完成**",
        "\n注：R1 的方向一致性实测 13/15 恰好等于通过门槛（≥13/15），属**临界通过**，特此标注。\n\n至此 **S6–S8 全部完成**",
        "e34-borderline")

# e35 §3.9 发现 N 措辞
s = rep(s, "**初步发现 1（结构轴）", "**发现 1（结构轴）", "e35-f1")
s = rep(s, "**初步发现 2（活性轴）", "**发现 2（活性轴）", "e35-f2")
s = rep(s, "**初步发现 3（谁能救）", "**发现 3（谁能救）", "e35-f3")
s = rep(s, "**初步发现 4（与距离轴正交", "**发现 4（与距离轴正交", "e35-f4")

# e36 发现2/3 数字口径标注
s = rep(s, "活性偏移下边际覆盖降到 **0.188（40%）/ 0.284（20%）**",
        "活性偏移下边际覆盖降到 **0.188（偏移 40%）/ 0.284（偏移 20%）**（全体分子，非悬崖组）",
        "e36-f2-label")
s = rep(s, "act_shift40 下 +10.4 pp\n（0.285 vs 0.181，p=1.2×10⁻⁴，15/15）",
        "act_shift40 下悬崖组 +10.4 pp\n（0.285 vs 0.181，p=1.2×10⁻⁴，15/15）",
        "e36-f3-label-a")
s = rep(s, "act_shift20 下 +16.0 pp（0.439 vs 0.279）",
        "act_shift20 下 +16.0 pp（悬崖组 0.439 vs 0.279）",
        "e36-f3-label-b")

# e37 发现4 18.1 分层口径消歧（正文 + 图14C caption）
s = rep(s, "缩小**（18.1 → 17.4 pp），且",
        "缩小**（最疏十分位 18.1 → 最密十分位 17.4 pp，分层口径），且", "e37-f4-body")
s = rep(s, "差距不缩小（18.1 → 17.4 pp）", "差距不缩小（最疏 18.1 → 最密 17.4 pp）", "e37-f4-cap")

# e38 §3.10 [50] 划界句
s = rep(s, "（BBBP 少数类覆盖跌至 64.8%、ClinTox 跌至 4.2%，类条件化可修复；Tursunbadalov 等[50]）。",
        "（BBBP 少数类覆盖跌至 64.8%、ClinTox 跌至 4.2%，类条件化可修复；Tursunbadalov 等[50]）。"
        "但两文须划清边界：[50] 的类条件化按**校准集真实标签**分组（部署时合法，校准集本有标签），修复的是**少数类**子群；"
        "本文检验的是按**预测类**分组的 Mondrian（不使用任何标签信息），在二分类下分箱分位数与全局几乎重合。"
        "两文同为\"边际达标、子群塌陷\"，但机制不同——类别不平衡 vs 局部标签矛盾——结论不可互推。",
        "e38-s310-delimit")

# e39 数据可用性：许可与仓库
s = rep(s, "MoleculeACE 基准数据来自 van Tilborg 等[25]（CC-BY 4.0）；分类基准来自 MoleculeNet（CC-BY）。",
        "MoleculeACE 基准（代码与整理数据，MIT 许可）来自 van Tilborg 等[25]，底层数据来自 ChEMBL（CC BY-SA）；分类基准来自 MoleculeNet（CC-BY）。",
        "e39-license")
s = rep(s, "本文全部分析脚本、逐数据集/逐种子中间结果表与分子级记录（70,170 行）随代码仓库发布，",
        "本文全部分析脚本、逐数据集/逐种子中间结果表与分子级记录（70,170 行）随代码仓库发布"
        "（仓库在论文接收后公开并附 DOI；审稿期间以补充材料提供），",
        "e39-repo")

# ============ 第二部分：全文引用重编号 ============
ORDER = [18, 13, 14, 47, 15, 19, 48, 3, 4, 5, 20, 21, 23, 45, 9, 10, 11,
         25, 26, 28, 27, 29, 30, 22, 33, 37, 32, 24, 34, 1, 2, 6, 8, 7, 12,
         31, 35, 36, 16, 17, 40, 41, 39, 42, 43, 38, 44, 49, 46, 50]
assert sorted(ORDER) == list(range(1, 51)), "ORDER must be a permutation of 1..50"
mapping = {old: i + 1 for i, old in enumerate(ORDER)}

TOK = r"\[(\d{1,2}(?:\s*[,，]\s*\d{1,2})*)\]"


def flatten(text):
    out = []
    for t in re.findall(TOK, text):
        out.extend(int(x) for x in re.split(r"[,，]", t))
    return out


before = flatten(s)
assert sorted(set(before)) == list(range(1, 51)), f"pre-renumber citation set: {sorted(set(before))}"


def remap(m):
    nums = [int(x) for x in re.split(r"[,，]", m.group(1))]
    return "[" + ",".join(str(mapping[n]) for n in sorted(nums, key=lambda v: mapping[v])) + "]"


s2 = re.sub(TOK, remap, s)
after = flatten(s2)
assert sorted(mapping[v] for v in before) == sorted(after), "renumber multiset mismatch"
first_seen = []
for v in after:
    if v not in first_seen:
        first_seen.append(v)
assert first_seen == sorted(first_seen), f"post-renumber order not monotonic: {first_seen}"
TAGS.append(f"renumber({len(before)} tokens)")
s = s2

save(p, s)

# ============ 第三部分：参考文献.md 重排 + 作者全名 ============
rp = "参考文献.md"
r = load(rp)
old_auth = "Tursunbadalov M, Tursunbadalov M."
assert r.count(old_auth) == 1, r.count(old_auth)
r = r.replace(old_auth, "Tursunbadalov Muhammadjon, Tursunbadalov Mustafojon.")

rlines = r.split("\n")
starts = [i for i, l in enumerate(rlines) if re.match(r"^\[\d{1,2}\]", l)]
assert len(starts) == 50, f"entry count {len(starts)}"
head = rlines[: starts[0]]
blocks = {}
bounds = []
for k, i in enumerate(starts):
    j = i + 1
    while j < len(rlines) and not re.match(r"^\[\d{1,2}\]", rlines[j]) and not rlines[j].startswith("#"):
        j += 1
    bounds.append((i, j))
    old = int(re.match(r"^\[(\d{1,2})\]", rlines[i]).group(1))
    blocks[old] = rlines[i:j]
assert set(blocks.keys()) == set(range(1, 51)), sorted(set(blocks.keys()) ^ set(range(1, 51)))
trailer = rlines[bounds[-1][1]:]

out = list(head)
for new_num in range(1, 51):
    body = list(blocks[ORDER[new_num - 1]])
    body[0] = re.sub(r"^\[\d{1,2}\]", f"[{new_num}]", body[0], count=1)
    out.extend(body)
out.extend(trailer)
save(rp, "\n".join(out))
TAGS.append("refs-reordered+relabelled")

# 校验：参考文献编号顺序 1..50
r2 = load(rp)
labels = [int(m.group(1)) for m in re.finditer(r"^\[(\d{1,2})\]", r2, re.M)]
assert labels == list(range(1, 51)), labels[:10]
assert r2.count("Tursunbadalov Muhammadjon") == 1
assert "[30] Vovk V, Gammerman A, Shafer G." in r2
assert "[1]" in r2 and "Algorithms" in r2
TAGS.append("refs-order-verified")

# ============ 第四部分：构建脚本横幅 ============
bp = "scripts/build_paper_html.py"
b = load(bp)
assert b.count("初稿 v4") == 1, b.count("初稿 v4")
b = b.replace("初稿 v4", "初稿 v5")
save(bp, b)
TAGS.append("banner-v5")

# ============ 最终回读验证 ============
final = load(p)
stray = re.findall(r"图 \d+、\d", final)
assert not stray, f"stray figure refs: {stray}"
for probe, expect in [("初步发现", 0), ("d_pairgap", 0), ("R1 ——", 0), ("差距，§3.4", 0),
                      ("富集 1.5–2.4 倍", 3), ("临界通过", 1), ("未在 OSF/AsPredicted 公开注册", 1),
                      ("net_mag（无量纲比值）", 1), ("方向不一致", 1), ("按强度递增的六级偏移强度谱", 1),
                      ("图 9、图 10", 1), ("图 11、图 12", 1), ("MIT 许可", 1),
                      ("校准集真实标签", 1), ("统计单位约定", 1), ("Venn–ABERS", 1)]:
    got = final.count(probe)
    assert got == expect, f"probe {probe!r}: got {got}, expect {expect}"
final_tokens = flatten(final)
first2 = []
for v in final_tokens:
    if v not in first2:
        first2.append(v)
assert first2 == list(range(1, 51)), f"final citation order: {first2}"

print(f"ALL {len(TAGS)} EDIT GROUPS OK")
print("tags:", ", ".join(TAGS))
