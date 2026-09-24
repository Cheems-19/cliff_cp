# -*- coding: utf-8 -*-
"""修订轮 3：按三份修改建议（建议1/2/3）对 论文全文_v1.md + 参考文献.md + SI_v1.md 做确定性编辑。

纪律：
- CRLF 夹层（读入统一为 \n，写回还原为 \r\n）
- 每个 rep 断言 anchor count == 1
- 图号重排在所有内容编辑之前做（一次正则遍历，无级联）
- 新增文本中的图号一律直接写【重排后的新编号】
"""
import io
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(path):
    with io.open(os.path.join(ROOT, path), "r", encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


def save(path, s):
    with io.open(os.path.join(ROOT, path), "w", encoding="utf-8", newline="") as fh:
        fh.write(s.replace("\n", "\r\n"))


def rep(s, old, new, tag):
    n = s.count(old)
    assert n == 1, "[%s] anchor count=%d" % (tag, n)
    return s.replace(old, new)


# ============================================================
# Part 1: 论文全文_v1.md
# ============================================================
s = load("论文全文_v1.md")

# ---- 1a. 图号重排（old->new：14->1, 1->2, 2->3, 10->4, 12->5, 3->6, 4->7,
#             9->8, 5->9, 6->10, 7->11, 8->12, 11->13, 13->14, 15->15）----
FIGMAP = {14: 1, 1: 2, 2: 3, 10: 4, 12: 5, 3: 6, 4: 7, 9: 8,
          5: 9, 6: 10, 7: 11, 8: 12, 11: 13, 13: 14, 15: 15}


def _fig_sub(m):
    has_space = m.group(1) == " "
    return ("图 " if has_space else "图") + str(FIGMAP[int(m.group(2))])


s = re.sub(r"图( ?)(\d{1,2})", _fig_sub, s)

# ---- 1b. 标题行 v3 -> v4 ----
s = rep(s, "（初稿 v3 · 2026-09-24）", "（初稿 v4 · 2026-09-24）", "title")

# ---- 1c. §3.1 标题图号顺序修正（重排后为「图 2、图 1」，按出现顺序应为 1、2）----
s = rep(s, "且不随校准粒度消失（图 2、图 1）", "且不随校准粒度消失（图 1、图 2）", "sec31-title")

# ---- 1d. §3.3 标题补图 6（旧图3=消融此前缺正文编号引用）----
s = rep(s, "部署前判据与同等覆盖率下的安全信号（图 5、表 2、图 7）",
        "部署前判据与同等覆盖率下的安全信号（图 5、表 2、图 6、图 7）", "sec33-title")

# ---- 1e. 摘要段 1 重写（P0：p 值对齐 2.1e-3；双口径 cliffB/cliffA；
#         移除 §3.9 探索性数字；偏移谱定性句；分类三集口径）----
OLD_ABS1 = ('共形预测（conformal prediction）为分子性质预测提供有限样本的边际覆盖保证，但"边际"意味着保证只在平均意义上成立——'
            '共形失效在化学空间中如何分布，此前未被量化。在 MoleculeACE 基准的 15 个 ChEMBL 活性数据集上'
            '（70,170 条测试记录，来自 10 个划分；3 个模型族 × 2 种表示 × 7 种阈值设定），'
            '我们证明共形失效强烈集中在活性悬崖：悬崖分子的覆盖率比非悬崖分子低 **18.2 个百分点**'
            '（10 个随机划分，逐数据集 7.7–32.4 pp，15/15 方向一致）；换用更严格的 Bemis–Murcko 骨架分组划分后，'
            '差距**仍然显著、方向一致 12/15，但幅度约减半（+8.9 pp，p=4×10⁻³）**——'
            '即它是数据与模型的性质，而不是随机划分的产物，同时其**幅度**确有划分依赖，'
            '本文所有幅度结论均附划分标注；而在活性偏移划分（测试集偏向更高效力）下，'
            '名义保证整体失效（覆盖 0.19–0.28），仅自适应共形部分恢复（+10~16 pp，仍不足名义的一半）；'
            '在最强的真活性分子中，共形区间把 **35% 悬崖分子**的上界压在真实效力之下，而非悬崖分子仅 13%。'
            '该差距不是 α=0.10 的单点现象（α=0.07→0.20 时 +13.2→+30.1 pp，单调增长），'
            '也不局限于回归：在五个分类基准上，标签悬崖（训练邻居标签冲突）处的集合预测覆盖跌至 0.03–0.58，'
            '其中三集在 10/10 种子上方向一致、另两集 5/10–7/10，而按预测类做类条件化毫无帮助。')
NEW_ABS1 = ('共形预测（conformal prediction）为分子性质预测提供有限样本的边际覆盖保证，但"边际"意味着保证只在平均意义上成立——'
            '共形失效在化学空间中如何分布，此前未被量化。在 MoleculeACE 基准的 15 个 ChEMBL 活性数据集上'
            '（主分析 70,170 条测试记录；3 个模型族 × 2 种表示），我们证明共形失效强烈集中在活性悬崖：'
            '悬崖分子的覆盖率比非悬崖分子低 **18.2 个百分点**（机制定义 cliffB，10 个随机划分，逐数据集 7.7–32.4 pp，'
            '15/15 方向一致；数据集级定义 cliffA 为 9.5 pp）；换用 Bemis–Murcko 骨架分组划分后，'
            '差距**仍然显著但幅度约减半（cliffB +8.9 pp，p=2.1×10⁻³，12/15；cliffA 5.0 pp，13/15）**——'
            '它是数据与模型的性质，而非随机划分的产物；其幅度确有划分依赖，本文所有幅度结论均附划分标注。'
            '把两个划分点扩展为六级偏移强度谱后：结构偏移只收窄差距而不破坏名义保证；'
            '活性偏移（测试集偏向更高效力）则整体击穿名义保证。'
            '该差距不是 α=0.10 的单点现象（α=0.07→0.20 单调增长，+13.2→+30.1 pp），也不局限于回归：'
            '在五个分类基准上，标签悬崖（训练邻居标签冲突）处的集合预测覆盖跌至 0.03–0.58，'
            '方向在 BACE/HIV/Tox21-NR-AR 三集 10/10 种子上一致（另两集悬崖群过小，仅具趋势性），'
            '而按预测类做类条件化毫无帮助；在最强的真活性分子中，共形区间把 **35% 悬崖分子**的上界压在真实效力之下，'
            '而非悬崖分子仅 13%。')
s = rep(s, OLD_ABS1, NEW_ABS1, "abstract-1")

# ---- 1f. 摘要段 2：8.6 口径标签 ----
s = rep(s, "我们并不提出新的区间构造方法（CQR 之后仍残余 8.6 pp 差距）。",
        "我们并不提出新的区间构造方法（覆盖率匹配口径下，CQR 之后仍残余 8.6 pp 差距，§3.4）。",
        "abstract-2")

# ---- 1g. §1.3 Svensson 前作插入 ----
s = rep(s, "此后 CP 在 QSAR 中的应用持续增长：Xu 等[15] 为 RF/DNN/GBM 定制 CP 算法",
        "此后 CP 在 QSAR 中的应用持续增长：Svensson 等[47] 把 CP 回归用于 QSAR 并系统量化预测不确定性；\n"
        "Xu 等[15] 为 RF/DNN/GBM 定制 CP 算法", "svensson")

# ---- 1h. §1.3 Mondrian 应用先例（Sun 2017）----
s = rep(s, "**分组条件化及其已知局限。** Mondrian 条件化是追求组内覆盖的标准工具，",
        "**分组条件化及其已知局限。** Mondrian 条件化是追求组内覆盖的标准工具，\n"
        "在大型不平衡生物活性数据集上已有应用先例（Sun 等[48] 的 Mondrian cross-conformal），", "sun2017")

# ---- 1i. §2.1 定义-指标耦合段（B-2）----
s = rep(s, "**划分**：主分析用随机划分 train/cal/test = 60/20/20，重复 10 个种子",
        "**定义与指标的耦合**：三种悬崖定义与各节读数一一对应——主读数用 `cliffB`（机制定义：\n"
        "标签较高侧，直接刻画\"预测最易失败的族群\"），`cliffA`（数据集级、与划分协议无关）作为\n"
        "预注册稳定读数（§3.8），`strong cliff` 仅用于敏感性检查；后文所有\"悬崖组覆盖\"均指按\n"
        "对应定义筛出的子群覆盖率，两套定义不可互换（§3.8 表 4 并列报告两口径）。\n"
        "\n"
        "**划分**：主分析用随机划分 train/cal/test = 60/20/20，重复 10 个种子", "b2-coupling")

# ---- 1j. 表 1：8 个实现、7 个独立量 + d_pair_gap/d_pairgap 合并 ----
s = rep(s, "**条件化信号（8 个）**：全部只依赖训练集与查询分子的结构/标签分布，",
        "**条件化信号（8 个实现、7 个独立量）**：全部只依赖训练集与查询分子的结构/标签分布，", "t1-count")
s = rep(s, "| `d_pair_gap` | 邻域内最大标签差 | 数据侧 |\n"
           "| `d_pairgap` | 同上（另一实现） | 数据侧 |",
        "| `d_pair_gap`（另实现记 `d_pairgap`） | 邻域内最大标签差（同一量的两种实现） | 数据侧 |",
        "t1-merge")

# ---- 1k. 公式 (1) Unicode 下标 ----
s = rep(s, "`s_i = |y_i − ŷ_i|` (1)", "`sᵢ = |yᵢ − ŷᵢ|` (1)", "eq1")

# ---- 1l. 公式 (3) + §2.3 A/B 列表改词 ----
s = rep(s, "- **A（看见悬崖）**：`s` → 是否为悬崖分子；\n"
           "- **B（预测失效）**：`s` → 该分子的区间是否未覆盖真实标签。\n"
           "\n"
           "**错配量**\n"
           "\n"
           "`M(s) = AUC_B − AUC_A`  (3)",
        "- **看见悬崖（cliff）**：`s` → 是否为悬崖分子；\n"
        "- **预测失效（fail）**：`s` → 该分子的区间是否未覆盖真实标签。\n"
        "\n"
        "**错配量**\n"
        "\n"
        "`M(s) = AUC_fail(s) − AUC_cliff(s)`  (3)", "eq3")

# ---- 1m. 公式 (4)：q_{k(i)} -> q_k，ε_i -> εᵢ ----
s = rep(s, "**命题 1（覆盖转移恒等式）** 记测试分子 `i` 的非一致性分数 `ε_i = |y_i − ŷ_i|`，\n"
           "全局分位数为 `q̄`，按信号 `s` 分箱后 `i` 所在箱的分位数为 `q_{k(i)}`。则对任意子群 `G`：\n"
           "\n"
           "`ΔCov_G := Cov_G(Mondrian_s) − Cov_G(global)`\n"
           "`= (1/|G|) · [ #{i∈G : q̄ < ε_i ≤ q_{k(i)}} − #{i∈G : q_{k(i)} < ε_i ≤ q̄} ]`  (4)",
        "**命题 1（覆盖转移恒等式）** 记测试分子 `i` 的非一致性分数 `εᵢ = |yᵢ − ŷᵢ|`，\n"
        "全局分位数为 `q̄`，按信号 `s` 分箱后 `i` 所在箱 `k` 的分位数记为 `q_k`。则对任意子群 `G`：\n"
        "\n"
        "`ΔCov_G := Cov_G(Mondrian_s) − Cov_G(global)`\n"
        "`= (1/|G|) · [ #{i∈G : q̄ < εᵢ ≤ q_k} − #{i∈G : q_k < εᵢ ≤ q̄} ]`  (4)", "eq4")

# ---- 1n. 公式 (5)：q_{k(i)} -> q_k ----
s = rep(s, "`net_mag(G; s) = E_{i∈G}[ max(0, (q_{k(i)} − q̄)/q̄) ] − E_{i∈G}[ max(0, (q̄ − q_{k(i)})/q̄) ]`  (5)",
        "`net_mag(G; s) = E_{i∈G}[ max(0, (q_k − q̄)/q̄) ] − E_{i∈G}[ max(0, (q̄ − q_k)/q̄) ]`  (5)", "eq5")
s = rep(s, "`q_{k(i)}` 与 `q̄` 只依赖校准集", "`q_k` 与 `q̄` 只依赖校准集", "eq5b")

# ---- 1o. §2.8 瘦身（脚本路径移 SI §S11）----
s = rep(s, "- 所有分析脚本在 `scripts/`，核心库在 `src/cliffcp/`；\n"
           "  分子级记录落盘（70,170 行）供二次分析。",
        "- 脚本组织、结果文件索引与分子级记录（70,170 行）见 SI §S11。", "s28-slim")
s = rep(s, "- 全部随机种子固定；每个表格可由脚本一键重生成。",
        "- 全部随机种子固定；每个表格可由脚本一键重生成（脚本组织见 SI §S11）。", "s28-slim2")

# ---- 1p. §2 末加 2.9 口径速查 + 符号表 ----
OLD_S29_ANCHOR = "- 全部随机种子固定；每个表格可由脚本一键重生成（脚本组织见 SI §S11）。\n\n---\n\n## 3 结果"
NEW_S29 = """- 全部随机种子固定；每个表格可由脚本一键重生成（脚本组织见 SI §S11）。

### 2.9 口径速查与符号表

本文的关键数字分属不同「划分协议 × 悬崖定义 × 归一化口径」组合，为避免混用，集中列示：

| 数字 | 口径 | 出处 |
|---|---|---|
| 18.2 pp | 随机划分 · cliffB 机制定义 · α=0.10 · 原始覆盖率 | §3.1 |
| 17.9 pp | 同上，插值到边际覆盖 0.90（方法阶梯口径） | §3.4 |
| 17.3 pp | 随机划分 · cliffB · 5 种子（表 4 global 参照） | §3.8 |
| 19.8 pp | 随机划分 · cliffB · 3 种子（偏移谱基准档） | §3.9 |
| 9.5 / 5.0 pp | cliffA 数据集级 · 随机 / 骨架划分 | §3.8 |
| 8.9 pp | 骨架划分 · cliffB 机制定义 | §3.8 |
| +1.22 pp | `d_wstd` · 覆盖率匹配口径（插值到边际覆盖 0.90） | §3.3 表 2 |
| +0.36 / −0.71 pp | `d_wstd` · 原始覆盖率 · 随机 / 骨架 | §3.8 表 4 |
| 8.6 pp | CQR 后残余差距 · 覆盖率匹配口径 · cliffB | §3.4 表 3 |
| 6.8 pp | CQR 后残余差距 · 原始覆盖率 · cliffA · 随机划分（9.5 − 2.73） | §3.8 |
| 3.6–5.3 倍 | AD 域内悬崖/非悬崖失败率比 · 8 种界定逐项 | §3.5 |
| 4.2 倍 | 主界定（距离型 20% 分位）的同一比值 | §3.5 |

**符号表**：`α` 目标失覆盖率（主配置 0.10）；`εᵢ` 分子 i 的非一致性分数；`q̂` 校准分位数（global）；`q̄` 全局分位数；`q_k` 信号 `s` 分箱后箱 `k` 的校准分位数；`ΔCov_G` 子群 `G` 的覆盖变化（Mondrian − global）；`M(s)` 错配量 = AUC_fail − AUC_cliff（需测试标签，事后）；`net_mag(G; s)` 净加宽幅度（免测试标签，部署前）；`cliffA`/`cliffB` 悬崖的两种定义（数据集级 / 机制定义）；`n_bins` Mondrian 分箱数。

---

## 3 结果"""
s = rep(s, OLD_S29_ANCHOR, NEW_S29, "sec29")

# ---- 1q. §3.2 图4 处删脚本名 ----
s = rep(s, "（图 4，`scripts/analyze_misalignment_relation.py`）", "（图 4）", "fig4-script")

# ---- 1r. §3.2 错配量重复定义精简（M(s) 三遍 -> 一遍）----
s = rep(s, "**错配量**定义为 `M(s) = AUC(s 预测失效) − AUC(s 看见悬崖)`。\n"
           "`M>0` 的信号\"看得见失效却看不见悬崖\"，`M≈0` 表示两种能力匹配，\n"
           "`M<0` 表示反向偏科。\n"
           "\n"
           "按 8 个信号 × 4 个模型/表示族共 28 个 (族, 信号) 点\n"
           "（图 4）：",
        "按 §2.3 的定义（`M(s) = AUC_fail(s) − AUC_cliff(s)`；`M>0` 表示\"看得见失效却看不见悬崖\"）。\n"
        "\n"
        "按 7 个独立信号（`d_pair` 与 `d_pair_gap` 为同一量的两种实现，故 8 个实现 × 4 族共 32 点、"
        "去重后 28 点；图 4）逐族扫描：", "ms-def")

# ---- 1s. §3.2 8 信号清单表格化 + 28/32 + rf_std 口径注 ----
OLD_LIST = ("把全部 8 个信号放在一起看，可以精确刻画这个反例的边界（M 取四族范围；net_mag 与实测 ΔCov 为 RF 族、"
            "两个表示的均值，即图 5 的数据）：`d_pair`/`d_pair_gap` 的 M = −0.19~−0.18、net_mag ≈ 0（−0.004）、"
            "实测 −0.3/−0.6 pp；`d_iqr` 的 M = −0.04~−0.02、net_mag +0.033、实测 +0.5 pp；"
            "`d_wstd`/`d_std` 的 M ≈ −0.03~0.00、net_mag +0.063/+0.060、实测 +1.7/+1.9 pp；"
            "`d_nb` 的 M = +0.14~+0.16、net_mag −0.046、实测 −3.6 pp；"
            "`rf_std` 的 M = +0.05~+0.18（族依赖：D-MPNN 族仅 +0.054，该族伤害也接近消失，+0.4 pp）、"
            "net_mag 在两表示间为 −0.033~+0.001、实测 −5.2/−6.5 pp；"
            "`d_nn_sim` 的 M = +0.40~+0.47、net_mag +0.037、实测 +1.7 pp。据此，"
            "**落入「M>0 但净加宽」区域的信号只有一个：`d_nn_sim`**；M>0 的另外两个信号（`d_nb`、`rf_std`）"
            "都落在净收紧（或中性偏收紧）方向且显著伤害。`d_nn_sim` 为何落入该区已有直接检验："
            "**分箱约定的产物**——无邻居分子经 `np.digitize` 全部落入顶箱、抬高顶箱分位数，"
            "恰好保护了同箱的悬崖分子（若按 NaN→0 落箱，同一信号约 −12 pp），见 §4.3 条 13。")
NEW_LIST = ("错配量 M 与伤害的逐信号对应关系汇总如下（M 取四族范围；net_mag 与实测 ΔCov 为 RF 族"
            "两个表示的均值，即图 5 的数据；信号级合并口径见注）：" + "\n\n" +
            "| 信号 | M（四族范围） | net_mag (pp) | 实测 ΔCov (pp) | 判读 |\n"
            "|---|---|---|---|---|\n"
            "| `d_pair` / `d_pair_gap` | −0.19 ~ −0.18 | ≈ 0（−0.004） | −0.3 / −0.6 | 反向偏科，不伤害 |\n"
            "| `d_iqr` | −0.04 ~ −0.02 | +0.033 | +0.5 | 无判别力，接近零 |\n"
            "| `d_wstd` / `d_std` | ≈ −0.03 ~ 0.00 | +0.063 / +0.060 | +1.7 / +1.9 | 对齐信号，小幅受益 |\n"
            "| `d_nb` | +0.14 ~ +0.16 | −0.046 | −3.6 | 正向偏科，显著伤害 |\n"
            "| `rf_std` | +0.05 ~ +0.18（族依赖：D-MPNN 族仅 +0.054，该族伤害接近消失，+0.4 pp） "
            "| −0.033 ~ +0.001（实现级）\\* | −5.2 / −6.5 | 正向偏科，显著伤害 |\n"
            "| `d_nn_sim` | +0.40 ~ +0.47 | +0.037 | +1.7 | **唯一落入「M>0 但净加宽」区** |\n"
            "\n"
            "\\* `rf_std` 的实现级 net_mag（两表示分别 +0.001 / −0.033）与 SI §S5.5 合并判据的信号级 net_mag"
            "（−1.63，两表示平均）并不矛盾：前者是 RF 族逐实现的读数，后者把两表示合并；该信号的真实伤害来自"
            "\"大幅收紧而校准侧质量小\"，幅度被实现级 net_mag 低估，故 S5.5 以箱分位数比值（q_ratio）补足"
            "（§4.3 条 13、SI §S5.5）。\n"
            "\n"
            "M>0 的三个信号中，`d_nb` 与 `rf_std` 都落在净收紧（或中性偏收紧）方向且显著伤害；"
            "`d_nn_sim` 为何落入净加宽区已有直接检验：**分箱约定的产物**——无邻居分子经 `np.digitize` "
            "全部落入顶箱、抬高顶箱分位数，恰好保护了同箱的悬崖分子（若按 NaN→0 落箱，同一信号约 −12 pp），"
            "见 §4.3 条 13。")
s = rep(s, OLD_LIST, NEW_LIST, "sec32-table")

# ---- 1t. §3.3 范围修正（S55 单元级 −3.3~−6.5）----
s = rep(s, "净收紧的信号（`d_nb`、`rf_std`）预测为负且实测为负（−3.1 ~ −6.5 pp），",
        "净收紧的信号（`d_nb`、`rf_std`）预测为负且实测为负（−3.3 ~ −6.5 pp），", "range-fix")

# ---- 1u. §3.3 补图 6 / 图 7 正文引用 ----
s = rep(s, "每个 α 都是真实可用的共形方法，无需任何事后缩放。逐配对插值 + 配对 Wilcoxon（表 2）。",
        "每个 α 都是真实可用的共形方法，无需任何事后缩放。逐配对插值 + 配对 Wilcoxon（表 2）。\n"
        "两种归一化的逐信号对比见图 6（收益 vs 代价散点）与图 7（宽度–覆盖前沿的两种视图）。", "fig67-ref")

# ---- 1v. 阈值敏感性括号去重 ----
s = rep(s, "阈值敏感性（**7 配置**（完整表见 SI §S2）：sim × act × n_bins 单因子扫描，表见 SI §S2）显示：",
        "阈值敏感性（**7 配置**：sim × act × n_bins 单因子扫描，完整表见 SI §S2）显示：", "dup-paren")

# ---- 1w. §3.4 CQR 8.6 口径（两处）----
s = rep(s, "是最强的修复，把差距砍半，但全体分子付 6.3% 宽度代价",
        "是最强的修复，把差距砍半（残余 8.6 pp 为覆盖率匹配口径），但全体分子付 6.3% 宽度代价", "cqr-i")
s = rep(s, "CQR 的宽度代价与其后残余的 8.6 pp 差距，正是这一诊断的价值所在。",
        "CQR 的宽度代价与其后残余的 8.6 pp 差距（覆盖率匹配口径，表 3），正是这一诊断的价值所在。", "cqr-ii")

# ---- 1x. §3.5 AD 内分子占比基线 + 3.4 保守下界标注 ----
s = rep(s, "  共 8 种界定下**全部成立**；即使做最不利的跨界定配对（最低悬崖失败率 26.1% 对\n"
           "  最高非悬崖失败率 7.8%），比值仍为 3.4；",
        "  共 8 种界定下**全部成立**；即使做最不利的跨界定配对（最低悬崖失败率 26.1% 对\n"
        "  最高非悬崖失败率 7.8%），比值仍为 3.4（保守下界）。8 种界定下\"AD 内\"分子占测试集的\n"
        "  60.0%–89.8%，故各界定比较均在同界定内配对进行，不受分母规模差异影响；", "ad-base")

# ---- 1y. §3.5 召回分母标注 ----
s = rep(s, "配套的**无标签预警信号**（训练邻居标签离散度 `d_wstd` 的组内秩分位，\n"
           "不含查询标签）在 20% 人工复核预算下召回域内失效的 35.7%–38.9%\n"
           "（相对随机 1.78–1.95 倍，8/8 稳定）。",
        "配套的**无标签预警信号**（训练邻居标签离散度 `d_wstd` 的组内秩分位，\n"
        "不含查询标签）在 20% 人工复核预算下召回**域内失效**的 35.7%–38.9%\n"
        "（分母为域内失效分子，与下文「全部失效」分母不同，两套口径不可混用；\n"
        "相对随机 1.78–1.95 倍，8/8 稳定）。", "recall-denom")

# ---- 1z. §3.8 6.8 口径 ----
s = rep(s, "判读：**最优基线（CQR，随机划分）之后残余差距仍有 6.8 pp**（9.5 − 2.73），远超 R2 的 2 pp 门槛 → S7 成立。",
        "判读：**最优基线（CQR，随机划分）之后残余差距仍有 6.8 pp**（9.5 − 2.73；原始覆盖率口径、cliffA，"
        "与 §3.4 匹配口径的 8.6 pp 不同），远超 R2 的 2 pp 门槛 → S7 成立。", "gap68")

# ---- 1aa. R3 n=525 ----
s = rep(s, "逐行 +0.618，p = 1.1×10⁻⁵⁶），而 **AUC_cliffB 几乎不动**",
        "逐行 +0.618，p = 1.1×10⁻⁵⁶；n = 525 行 = 15 数据集 × 5 种子 × 7 档信号精度），"
        "而 **AUC_cliffB 几乎不动**", "n525")

# ---- 1ab. 预注册表述 ----
s = rep(s, "至此 **S6–S8 全部完成**，预注册的三个次要假设均有结论。",
        "至此 **S6–S8 全部完成**：两个预注册假设（S6/S7）通过判读标准，R3 为探索性受控检验，"
        "其结果与机制预言一致。", "prereg")

# ---- 1ac. §3.9 发现 -> 初步发现 ----
s = rep(s, "**发现 1（结构轴）：差距单调收窄，但始终显著。**",
        "**初步发现 1（结构轴）：差距单调收窄，但始终显著。**", "disc1")
s = rep(s, "**发现 2（活性轴）：边际保证被整体击穿。**",
        "**初步发现 2（活性轴）：边际保证被整体击穿。**", "disc2")
s = rep(s, "**发现 3（谁能救）：只有自适应共形有实质改善，但远不彻底。**",
        "**初步发现 3（谁能救）：只有自适应共形有实质改善，但远不彻底。**", "disc3")
s = rep(s, "**发现 4（与距离轴正交——本文主张的测量版）。**",
        "**初步发现 4（与距离轴正交——本文主张的测量版）。**", "disc4")

# ---- 1ad. Lee 2026 [49] 插入 ----
s = rep(s, "当测试集系统性偏向更高效力时，任何基于随机可交换性的校准都失效——这正是\"向更高效力区间外推\"\n"
           "场景下使用共形区间的实际风险。",
        "当测试集系统性偏向更高效力时，任何基于随机可交换性的校准都失效——这正是\"向更高效力区间外推\"\n"
        "场景下使用共形区间的实际风险（针对标签偏移的加权共形方案见 Lee 等[49]；"
        "其与本文的差异在于它假设偏移类型已知并据此加权，本文则先测出偏移强度谱再报告哪些保证存活）。", "lee49")

# ---- 1ae. §3.10 APS [46] ----
s = rep(s, "- **边缘基准 APS（Adaptive Prediction Sets）并未真正修复**：APS 把标签悬崖覆盖拉到",
        "- **边缘基准 APS（Adaptive Prediction Sets[46]）并未真正修复**：APS 把标签悬崖覆盖拉到", "aps46")

# ---- 1af. Tursunbadalov [50] ----
s = rep(s, "- 边际覆盖贴名义（0.903–0.909），与回归一致：失效是**子群性**的，不是全局性的。",
        "- 边际覆盖贴名义（0.903–0.909），与回归一致：失效是**子群性**的，不是全局性的——\n"
        "  这一\"边际达标、子群塌陷\"的模式与近期在不平衡虚拟筛选上的独立报告一致\n"
        "  （BBBP 少数类覆盖跌至 64.8%、ClinTox 跌至 4.2%，类条件化可修复；Tursunbadalov 等[50]）。", "tursun50")

# ---- 1ag. §3.10 注释精简（脚本/路径移 SI §S11）----
s = rep(s, "BACE/HIV/Tox21-NR-AR 在 10/10 种子上一致。\n"
           "脚本 `scripts/phase17_classify.py`，结果 `results/phase17_classify_v2/`；\n"
           "数据 MoleculeNet CC-BY，下载缓存于服务器 `data/classification/`。）",
        "BACE/HIV/Tox21-NR-AR 在 10/10 种子上一致。实现与数据缓存见 SI §S11。）", "s310-note")

# ---- 1ah. §4.3 条 1：8.6 口径 ----
s = rep(s, "   CQR 之后悬崖差距仍有 8.6 pp。域内预警是 AD 的补充，不是替代。",
        "   CQR 之后悬崖差距仍有 8.6 pp（覆盖率匹配口径，§3.4 表 3；原始覆盖率 cliffA 口径为 6.8 pp，§3.8）。"
        "域内预警是 AD 的补充，不是替代。", "lim1")

# ---- 1ai. §4.3 条 3：29/52 解释 ----
s = rep(s, "3. `d_wstd` 的覆盖率匹配改善 p=0.099，未过 0.05——只能表述为\n"
           "   \"小但方向一致（10/15 数据集、29/52 配对）\"。",
        "3. `d_wstd` 的覆盖率匹配改善 p=0.099，未过 0.05——只能表述为\n"
        "   \"小但方向一致（10/15 数据集、29/52 配对；52 为插值后非缺失的 (数据集, α) 配对总数\n"
        "   （15 数据集 × 4 档 α 去除 8 个悬崖分子过少的组合），29 为其中 Δ>0 者）\"。", "lim3")

# ---- 1aj. §4.3 条 11：靶点族聚类 ----
s = rep(s, "11. **异质性高（I² = 46%–88%）**：所有跨数据集合并量均报随机效应区间，\n"
           "   不报固定效应；逐数据集方向一致性作为辅助证据一并列出。",
        "11. **异质性高（I² = 46%–88%）**：所有跨数据集合并量均报随机效应区间，\n"
        "   不报固定效应；逐数据集方向一致性作为辅助证据一并列出。\n"
        "   此外 15 个 ChEMBL 基准**未按靶点族（如激酶/核受体/GPCR）聚类**，\n"
        "   同族数据集的效应可能相关，合并统计的有效自由度低于表面数据集数。", "lim11")

# ---- 1ak. §4.3 条 12：CQR 口径重写（裁决：CQR 在匹配扫描中已运行）----
s = rep(s, "12. 覆盖率匹配扫描未含 CQR（换 α 批跑使用 `--no-cqr` 以省 5 倍耗时）：\n"
           "   CQR 只在主配置 α=0.10 单独报告（表 3），未进入插值比较。",
        "12. CQR 的覆盖率匹配读数（表 3）来自单独的匹配批跑（对每个方法按各自 target α 插值到\n"
        "   边际覆盖 0.90）；换 α 的敏感性批跑（α ∈ {0.07, 0.13, 0.16, 0.20}）未含 CQR\n"
        "   （`--no-cqr`，其成本会把单数据集耗时从 1.3 min 拉到 6.7 min），故 CQR 的 α 敏感性未单独报告。", "lim12")

# ---- 1al. §4.3 条 13：脚本路径清理 ----
s = rep(s, "   **「为何落在净加宽一侧」已完成直接检验**（流水线内部真值落盘，脚本\n"
           "   `scripts/phase16_mechanism.py`，结果 `results_server/PHASE16_MECHANISM.txt`）：",
        "   **「为何落在净加宽一侧」已完成直接检验**（流水线内部真值逐箱落盘；脚本与结果文件见 SI §S11）：", "lim13")

# ---- 1am. §4.4 重写 ----
OLD_44 = ("1. 把 D-MPNN 集成成员加到 8–10 个，检验 `rf_std` 通道\n"
          "   （需固定模型只改信号，避免本文 §3.6 指出的混杂）。\n"
          "2. 把\"错配量 → 伤害\"从甄别规则升级为可预测的定量关系\n"
          "   ——已完成：`d_nn_sim` 反例已由分箱约定解释（§4.3 条 13），\n"
          "   且 `net_mag`（幅度）与 q 比（机制）在信号级符号一致 8/8、单元级 11/13\n"
          "   （2 个例外均落在「无实质影响」区），二者并用无一漏报（SI §S5.5）。\n"
          "3. 覆盖率检验的正式版（Kupiec/Christoffersen）在偏移强度谱上的行为。\n"
          "4. 划分升级为偏移强度谱，替代单一 scaffold/随机划分的质疑。")
NEW_44 = ("1. 覆盖率检验的正式版（Kupiec/Christoffersen）在偏移强度谱上的行为。\n"
          "2. 与\"悬崖感知\"的评估基线（如对悬崖分子显式加权的校准/排序基线，CRC 类方法）\n"
          "   在同一把尺子上比较：本文的诊断框架回答\"哪些信号危险\"，与之互补的问题是\n"
          "   \"显式利用悬崖信息能买回多少覆盖\"。\n"
          "3. 按靶点族（激酶/核受体/GPCR 等）对 15 个数据集聚类，检验错配甄别规则与\n"
          "   合并统计的跨族稳定性（§4.3 条 11）。\n"
          "4. 把\"错配量 → 伤害\"的定量关系做跨表示/跨模型族的元回归：甄别规则（判类别、判方向）\n"
          "   已闭环（§3.2、SI §S5.5），幅度预测仍需逐场景实证检验（§3.3）。")
s = rep(s, OLD_44, NEW_44, "sec44")

# ---- 1an. 数据可用性 + COI；删图表清单 ----
OLD_TAIL_ANCHOR = "---\n\n## 图表清单（当前状态）"
NEW_TAIL = """---

## 数据与代码可用性

MoleculeACE 基准数据来自 van Tilborg 等[25]（CC-BY 4.0）；分类基准来自 MoleculeNet（CC-BY）。
本文全部分析脚本、逐数据集/逐种子中间结果表与分子级记录（70,170 行）随代码仓库发布，
文中全部表格与图均可由脚本一键重生成；脚本组织与结果文件索引见 SI §S11。

## 利益冲突声明

作者声明不存在利益冲突。"""
i = s.find(OLD_TAIL_ANCHOR)
assert i > 0, "[tail] anchor not found"
s = s[:i] + NEW_TAIL + "\n"
assert "图表清单" not in s, "[tail] checklist residue"

save("论文全文_v1.md", s)
print("[ok] 论文全文_v1.md: %d chars, %d CRLF lines" % (len(s), s.count("\n")))

# ============================================================
# Part 2: 参考文献.md
# ============================================================
r = load("参考文献.md")

r = rep(r, "> **全部 45 条已于 2026-09-23 逐条核实完毕（含 5 条 2026 年新文献的题名/作者更正）**，不再有【待核】条目。",
        "> 全部 50 条中 45 条已于 2026-09-23 逐条核实；新增 [46]–[50] 五条已于 2026-09-24 经检索核实。不再有【待核】条目。",
        "ref-head")

# 删自用注释
r = rep(r, "（适用域原则，Principle 3）", "", "ref16")
r = rep(r, "（X-/y-/Model-outlier 三类离群点框架）", "", "ref18")
r = rep(r, "（引言原文已逐字核验：\"a compound may lie well within the training domain yet be poorly predicted due to activity cliffs, noise in the training data, or model overfitting.\"）", "", "ref19")
r = rep(r, "（832K 次评估 / 14K 配置 / 6 数据集 / 100 种子；Mondrian 使转人工差距 +143%，同时覆盖差距 −26%，跨数据集均 p<0.001）", "", "ref21")
r = rep(r, "（stat.ME；ADNI/OASIS-3 两队列 × 2 预测器 × 9 属性，68 个组合中 57 个欠覆盖；机制为 rarity 与 tail-heaviness，高风险亚组平均缺口 6.1 pp）", "", "ref22")
r = rep(r, "（**与本文命题 1 最接近的理论工作**：给出 pooled 校准的组间覆盖扭曲「守恒律」与下界，尺度由跨组分位数异质性决定；并证明 Equalized Coverage 与 Equalized Set Size 在组基础率不同时不可兼得）", "", "ref23")
r = rep(r, "（math.ST；给出最小化区间长度的训练–校准划分比例的解析刻画，覆盖保证保持）", "", "ref24")
r = rep(r, "（xv + 81 页；\"even within the applicability domain … may arise due to the activity cliff behavior\" 即本文出发点）", "", "ref30")
r = rep(r, "（stat.ME；把条件失覆盖作非渐近三分解——分数估计误差 + 有限样本校准误差 + **内在条件错配误差**；与本文错配量/命题 1 同源，须并列引用）", "", "ref45")

# 新增 [46]
r = rep(r, "[13] Sluijterman L, Cator E, Heskes T. How to evaluate uncertainty estimates in machine learning for regression? arXiv:2106.03395, 2021.",
        "[13] Sluijterman L, Cator E, Heskes T. How to evaluate uncertainty estimates in machine learning for regression? arXiv:2106.03395, 2021.\n"
        "\n"
        "[46] Romano Y, Sesia M, Candès E J. Classification with valid and adaptive coverage. "
        "Advances in Neural Information Processing Systems (NeurIPS), 2020, 33: 3581–3591.", "ref46")

# 新增 [47]-[50]（第二节 [19] 之后）
r = rep(r, "[19] Jeliazkova N, Kochev N, Iliev L, Jeliazkov V. Beyond heuristics: a model-agnostic framework for uncertainty quantification in QSAR via adaptive conformal prediction. Chemical Research in Toxicology, 2026, 39(7): 1357–1376. doi:10.1021/acs.chemrestox.6c00065. PMID 42324899.",
        "[19] Jeliazkova N, Kochev N, Iliev L, Jeliazkov V. Beyond heuristics: a model-agnostic framework for uncertainty quantification in QSAR via adaptive conformal prediction. Chemical Research in Toxicology, 2026, 39(7): 1357–1376. doi:10.1021/acs.chemrestox.6c00065. PMID 42324899.\n"
        "\n"
        "[47] Svensson F, Aniceto N, Norinder U, Cortes-Ciriano I, Spjuth O, Carlsson L, Bender A. "
        "Conformal regression for quantitative structure–activity relationship modeling—quantifying prediction "
        "uncertainty. Journal of Chemical Information and Modeling, 2018, 58(5): 1132–1140.\n"
        "\n"
        "[48] Sun J, Carlsson L, Ahlberg E, Norinder U, Engkvist O, Chen H. Applying Mondrian cross-conformal "
        "prediction to estimate prediction confidence on large imbalanced bioactivity data sets. "
        "Journal of Chemical Information and Modeling, 2017, 57(7): 1591–1598.\n"
        "\n"
        "[49] Lee H, Kim J, Jadamba E, Choi S, Shin H. Conformal prediction for molecular properties under label "
        "shift. arXiv:2608.17678, 2026.\n"
        "\n"
        "[50] Tursunbadalov M, Tursunbadalov M. A quiet failure in calibrated virtual screening: marginal conformal "
        "prediction under-covers the minority class, and a class-conditional fix recovers it. arXiv:2607.06605, 2026.",
        "ref47-50")

r = rep(r, "**无孤立条目**：全部 45 条均在正文被引用（已验证）。",
        "**无孤立条目**：全部 50 条均在正文被引用（[46]–[50] 于 2026-09-24 插入正文 §1.3/§3.9/§3.10，已验证）。",
        "ref-orphan")

# 对照表加行（附内，不进交付物）
r = rep(r, "| §1.3 共形预测 × QSAR | [14] [15] [19]（另有\"近年在 ChEMBL 上大规模 CP 比较\"【待补引】） |",
        "| §1.3 共形预测 × QSAR | [14] [15] [19] [47] [48] |", "ref-map1")
r = rep(r, "| §1.3 活动悬崖 | [18] [22] [25] [26] [27] [28] [29] [30] |",
        "| §1.3 活动悬崖 | [18] [22] [25] [26] [27] [28] [29] [30] |\n"
        "| §3.9 标签偏移（加权共形） | [49] |\n"
        "| §3.10 APS / 少数类欠覆盖 | [46] [50] |", "ref-map2")

save("参考文献.md", r)
print("[ok] 参考文献.md: %d chars" % len(r))

# ============================================================
# Part 3: SI_v1.md —— 新增 S11
# ============================================================
si = load("SI_v1.md")

OLD_SI_END = "---\n\n*End of Supporting Information.*"
NEW_SI_END = """---

## S11  Implementation and data organization

- All analysis scripts live in `scripts/`, core library in `src/cliffcp/`; molecule-level records (70,170 rows) are dumped for secondary analysis.
- Misalignment-vs-harm panorama (figure source for the manuscript's misalignment scatter): `scripts/analyze_misalignment_relation.py`.
- Mechanism ground truth for the `d_nn_sim` binning artefact: `scripts/phase16_mechanism.py`, output `results_server/PHASE16_MECHANISM.txt` (per-bin quantile ratios; 321/321 no-neighbour molecules land in the top bin).
- Merged two-criterion analysis (manuscript §3.2 note and §4.4): `scripts/phase19_merged_criterion.py`, output `results_server/S55_MERGED_CRITERION.txt` (§S5.5).
- Classification extrapolation (manuscript §3.10): `scripts/phase17_classify.py`, results `results/phase17_classify_v2/`; MoleculeNet data (CC-BY) cached on the server under `data/classification/`.
- Coverage-matched paired statistics (manuscript §3.3/§3.4): `results_server/alpha_x/alpha_matched/COVERAGE_MATCHED_PAIRED.csv` (per (dataset, α) pairs after interpolation; `n_pairs` = 52 and `pos_pairs` = 29 for `mondrian_wstd`/`mondrian_std`).

---

*End of Supporting Information.*"""
si = rep(si, OLD_SI_END, NEW_SI_END, "si-s11")

save("SI_v1.md", si)
print("[ok] SI_v1.md: %d chars" % len(si))

# ============================================================
# 回读验证
# ============================================================
checks = [
    ("论文全文_v1.md", "p=2.1×10⁻³", 1),
    ("论文全文_v1.md", "n = 525 行 = 15 数据集 × 5 种子 × 7 档信号精度", 1),
    ("论文全文_v1.md", "### 2.9 口径速查与符号表", 1),
    ("论文全文_v1.md", "M(s) = AUC_fail(s) − AUC_cliff(s)", 2),   # §2.3 + §3.2
    ("论文全文_v1.md", "52 为插值后非缺失的 (数据集, α) 配对总数", 1),
    ("论文全文_v1.md", "未按靶点族（如激酶/核受体/GPCR）聚类", 1),
    ("论文全文_v1.md", "## 数据与代码可用性", 1),
    ("论文全文_v1.md", "## 利益冲突声明", 1),
    ("论文全文_v1.md", "Adaptive Prediction Sets[46]", 1),
    ("论文全文_v1.md", "Tursunbadalov 等[50]", 1),
    ("论文全文_v1.md", "Lee 等[49]", 1),
    ("论文全文_v1.md", "Sun 等[48]", 1),
    ("论文全文_v1.md", "Svensson 等[47]", 1),
    ("论文全文_v1.md", "图表清单", 0),
    ("论文全文_v1.md", "np.digitize", 2),   # §3.2 表格注后 + §4.3 条13
    ("论文全文_v1.md", "60.0%–89.8%", 1),
    ("论文全文_v1.md", "q_{k(i)}", 0),
    ("论文全文_v1.md", "q_{k(i)}".replace("k(i)", "k(i)"), 0),
    ("参考文献.md", "[46] Romano Y, Sesia M", 1),
    ("参考文献.md", "[50] Tursunbadalov M", 1),
    ("参考文献.md", "引言原文已逐字核验", 0),
    ("SI_v1.md", "## S11", 1),
    ("SI_v1.md", "COVERAGE_MATCHED_PAIRED.csv", 1),
]
ok = True
for path, needle, expect in checks:
    txt = load(path)
    got = txt.count(needle)
    status = "OK " if got == expect else "FAIL"
    if got != expect:
        ok = False
    print("%s %-18s %r -> %d (expect %d)" % (status, path, needle[:40], got, expect))

# 图号交叉验证：正文引用的图号与 caption 编号一一对应
p = load("论文全文_v1.md")
caps = re.findall(r"\*\*图 ?(\d+)\*\*", p)
print("caption 图号序列:", caps, "-> 期望 1..15")
assert caps == [str(i) for i in range(1, 16)], "caption 顺序错误"
imgs = re.findall(r"!\[图 ?(\d+)\]", p)
print("图片引用图号序列:", imgs)
assert imgs == [str(i) for i in range(1, 16)], "图片引用顺序错误"

# 旧图号残留检查（重排后不应再有「图 16」以上或不存在的引用）
bad = [m for m in re.findall(r"图 ?(\d+)", p) if int(m) > 15]
assert not bad, "非法图号: %s" % bad

print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
