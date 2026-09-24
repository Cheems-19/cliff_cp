# -*- coding: utf-8 -*-
import io

P = r"C:/Users/Administrator/Desktop/小论文/cliff_cp/论文全文_v1.md"
raw = io.open(P, encoding="utf-8").read()
print("has_crlf:", "\r\n" in raw)
s = raw.replace("\r\n", "\n")

repls = [
    # 1. 标题
    ("### 3.10 分类任务的外推（探索性）",
     "### 3.10 分类任务的外推"),

    # 2. 描述段（三 -> 五数据集，5 -> 10 种子）
    ("为检验现象不是回归设定特有，把「悬崖邻域失效」外推到三个分类基准（MoleculeNet：\nBBBP 1,975 / BACE 1,513 / Tox21-NR-AR 7,258 个分子；ECFP4 + 200 棵树的随机森林；\n随机划分 60/20/20 × 5 种子；集合预测共形，α=0.10）。分类里的悬崖类比是\n**标签悬崖**：测试分子存在训练邻居（Tanimoto ≥ 0.85）且标签不一致——稠密邻域中的标签矛盾。",
     "为检验现象不是回归设定特有，把「悬崖邻域失效」外推到五个分类基准（MoleculeNet：\nBBBP 1,975 / BACE 1,513 / ClinTox 1,461 / HIV 41,120 / Tox21-NR-AR 7,258 个分子；\nECFP4 + 200 棵树的随机森林；随机划分 60/20/20 × 10 种子；集合预测共形，α=0.10）。\n分类里的悬崖类比是**标签悬崖**：测试分子存在训练邻居（Tanimoto ≥ 0.85）且标签不一致\n——稠密邻域中的标签矛盾。"),

    # 3. 覆盖失效 bullet
    ("- **覆盖失效同样集中于标签悬崖**：BACE 0.644 vs 0.935（+29.1 pp，5/5）、\n  BBBP 0.467 vs 0.909（+44.2 pp，3/5）、Tox21-NR-AR **0.029 vs 0.906（+87.7 pp，5/5）**。\n  Wilcoxon p=0.0625 是 n=5 的最小可能值；标签悬崖群很小（4–27 个/种子），\n  故本节为**方向一致的探索性证据**，统计功效有限。",
     "- **覆盖失效同样集中于标签悬崖**：五个数据集的标签悬崖覆盖均远低于非悬崖，\n  差距 +35.7（BACE）至 +88.0 pp（Tox21-NR-AR）不等，方向在 BACE/HIV/Tox21-NR-AR 上\n  10/10、BBBP 7/10、ClinTox 5/10 一致；幅度显著大于回归设定下的 18.2 pp（cliffB），\n  说明分类中的标签矛盾比回归中的效力连续性断裂带来更尖锐的失效。"),

    # 4. 类条件化 bullet + 新增 APS bullet
    ("- **按预测类做类条件化（Mondrian）完全没有帮助**：三个数据集的覆盖与集合大小与 global\n  逐一相同——二分类下按预测类分箱的分位数与全局几乎重合。这把回归篇的结论\n  「条件化不是万能药、需要分区信号携带失效信息」原样带进了分类设定。",
     "- **按预测类做类条件化（Mondrian）完全没有帮助**：五个数据集的覆盖与集合大小与\n  global 逐一相同——二分类下按预测类分箱的分位数与全局几乎重合。相似度条件化\n  （mondrian_nnsim）仅有微小改善（cliff 覆盖最多 +5.3 pp，Tox21-NR-AR），远未弥合失效。\n  这把回归篇的结论「条件化不是万能药、需要分区信号携带失效信息」原样带进了分类设定。\n- **边缘基准 APS（Adaptive Prediction Sets）并未真正修复**：APS 把标签悬崖覆盖拉到\n  ≈1.0，但其集合大小也撑到 ≈2.0（二分类即包含全部标签），退化成「覆盖全标签的平凡解」——\n  gap 接近 0 是靠放弃所有判别力换来的，不构成对悬崖问题的实际解决。"),

    # 5. 边际覆盖 bullet + 脚注
    ("- 边际覆盖贴名义（0.905–0.913），与回归一致：失效是**子群性**的，不是全局性的。\n  Tox21 的悬崖覆盖近乎 0——邻居标签冲突的分子本质上就是模型学不到的分子，\n  与回归里 cliffB 的机制同源。\n\n（脚本 `scripts/phase17_classify.py`，结果 `results/phase17_classify/`；\n数据 MoleculeNet CC-BY，下载缓存于服务器 `data/classification/`。）",
     "- 边际覆盖贴名义（0.903–0.909），与回归一致：失效是**子群性**的，不是全局性的。\n  HIV 与 Tox21-NR-AR 的标签悬崖覆盖近乎 0（0.040 / 0.027）——邻居标签冲突的分子\n  本质上就是模型学不到的分子，与回归里 cliffB 的机制同源。\n\n（注：标签悬崖群较小（BBBP/ClinTox 仅 3–4 个/种子），故 BBBP 7/10、ClinTox 5/10 的\n方向一致主要体现趋势而非高功效结论；BACE/HIV/Tox21-NR-AR 在 10/10 种子上一致，功效充足。\n脚本 `scripts/phase17_classify.py`，结果 `results/phase17_classify_v2/`；\n数据 MoleculeNet CC-BY，下载缓存于服务器 `data/classification/`。）"),

    # 6. 摘要 B6 数字
    ("也不局限于回归：在三个分类基准上，标签悬崖（训练邻居标签冲突）处的集合预测覆盖跌至 0.03–0.64（方向 5/5），而按预测类做类条件化毫无帮助",
     "也不局限于回归：在五个分类基准上，标签悬崖（训练邻居标签冲突）处的集合预测覆盖跌至 0.03–0.58，其中三集在 10/10 种子上方向一致、另两集 5/10–7/10，而按预测类做类条件化毫无帮助"),
]

ok = 0
for i, (old, new) in enumerate(repls, 1):
    c = s.count(old)
    if c == 1:
        s = s.replace(old, new)
        ok += 1
        print(f"[repl {i}] OK")
    else:
        print(f"[repl {i}] WARN count={c} (expect 1) -> skipped")

# 写回
io.open(P, "w", encoding="utf-8").write(s)

# 立即读回验证
back = io.open(P, encoding="utf-8").read().replace("\r\n", "\n")
checks = ["### 3.10 分类任务的外推\n", "ClinTox 1,461", "+88.0 pp", "Adaptive Prediction Sets",
          "phase17_classify_v2", "0.03–0.58"]
print("---- verify in file ----")
for t in checks:
    print(("FOUND " if t in back else "MISSING ") + repr(t))
print("replaced:", ok, "/", len(repls))
