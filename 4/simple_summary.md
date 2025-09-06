总体思路（如何解决问题四）

目标：基于附件数据，判定胎儿是否存在 13/18/21 号染色体非整倍体异常。我们按“稳健可解释”的路线搭建了一个端到端的小流水线：

1) 数据清洗与特征工程

- 位置：[4/q4_female_processing.py](cci:7://file:///Users/Mac/Downloads/mm/4/q4_female_processing.py:0:0-0:0) 中的 [clean_female_sheet()](cci:1://file:///Users/Mac/Downloads/mm/4/q4_female_processing.py:94:0-207:18)
- 关键处理：
  - 合并异常标签：`is_abnormal`（任一 T13/T18/T21 为 1），保留 `ab_T13/ab_T18/ab_T21`
  - 离散化与指示变量：`IVF妊娠_编码`（自然=0，IVF=1），`高龄`（年龄≥35）
  - 标准化连续变量（z-score）：孕周、年龄、BMI、QC 相关（读段数、比对比例、过滤比例、GC 及各染色体 GC、X 浓度），但“Z 值本身不再标准化”
  - 输出文件：[4/female_cleaned.csv](cci:7://file:///Users/Mac/Downloads/mm/4/female_cleaned.csv:0:0-0:0)

2) 单因素 ANOVA 初筛（保留医学关键+放宽阈值）

- 位置：[4/q4_anova_feature_select.py](cci:7://file:///Users/Mac/Downloads/mm/4/q4_anova_feature_select.py:0:0-0:0)
- 原理：对每个数值特征分别做一次单因素方差分析（两组：正常/异常），使用 p 值（默认 0.10）或 eta² 作为筛选阈值；同时“始终保留 Z 值”以体现医学知识优先级
- 产物：
  - [4/female_anova_report.csv](cci:7://file:///Users/Mac/Downloads/mm/4/female_anova_report.csv:0:0-0:0)（每个特征的 F、p、eta² 等）
  - [4/female_cleaned_anova_selected.csv](cci:7://file:///Users/Mac/Downloads/mm/4/female_cleaned_anova_selected.csv:0:0-0:0)（被选特征+所有 Z 值）

3) 类别不平衡处理：ADASYN 过采样

- 位置：[4/q4_adasyn_oversample.py](cci:7://file:///Users/Mac/Downloads/mm/4/q4_adasyn_oversample.py:0:0-0:0) 的 [_adasyn_generate()](cci:1://file:///Users/Mac/Downloads/mm/4/q4_adasyn_oversample.py:41:0-127:34)
- 原理：在训练集上对少数类（异常）按“局部密度困难度”分配合成权重，靠近稀疏区域多合成，近邻少数类插值生成新样本（并先用中位数填补缺失避免邻居计算出错）
- 我们先做了目标占比 0.4 的版本，又针对优化做了目标占比 0.5 的版本：
  - 0.5 版本输出在：`4/sampling_050/`
  - 训练集少数类从 47 提升到 376，与多数类 376 达到 1:1

4) 训练与评估多模型（阈值可调）

- 位置：[4/models/q4_train_models.py](cci:7://file:///Users/Mac/Downloads/mm/4/models/q4_train_models.py:0:0-0:0)
- 模型：
  - 随机森林（带 5 折 F1 网格搜索CV）
  - XGBoost（已安装可用）
  - LightGBM（已安装可用）
- 统一前处理：中位数填补（Pipeline 中 `SimpleImputer(strategy="median")`），确保训练/评估一致
- 阈值策略：对模型输出“异常概率”采用固定阈值判定，高于阈值（如 0.7/0.5/0.3）报告预警

# “每一步”的原理为什么这样做

- 合并异常标签 + 保留多标签：统一任务目标（是否异常），又能在后续分任务分析各染色体类型。
- 标准化连续变量但“不标准化 Z 值”：Z 值本身就是标准化统计量（均值0方差1的意义），再标准化会破坏阈值的临床意义（如 |Z|≥3）。
- ANOVA 初筛：单变量维度快速过滤掉与目标相关性极低的变量，降低噪声和复杂度；同时“强制保留 Z 值”以承接医学常识。
- ADASYN：异常样本仅约 10%，若不处理，模型会偏向预测“正常”。ADASYN在稀疏区域更“用力”合成，有利于提升召回。
- 多模型齐上：树模型更适合结构化数据，可生成特征重要性；XGBoost/LightGBM对不平衡可设置 [scale_pos_weight](cci:1://file:///Users/Mac/Downloads/mm/4/models/q4_train_models.py:100:0-103:56)，性能通常不错。统一阈值便于反映“召回优先”的临床要求。

# 当前已有产物

- 特征工程输出：
  - [4/female_cleaned.csv](cci:7://file:///Users/Mac/Downloads/mm/4/female_cleaned.csv:0:0-0:0)（清洗+标准化）
  - [4/female_anova_report.csv](cci:7://file:///Users/Mac/Downloads/mm/4/female_anova_report.csv:0:0-0:0)（ANOVA 报告）
  - [4/female_cleaned_anova_selected.csv](cci:7://file:///Users/Mac/Downloads/mm/4/female_cleaned_anova_selected.csv:0:0-0:0)（筛选后的特征集）
- 过采样输出（目标占比 0.5）：
  - `4/sampling_050/female_train_original.csv`
  - `4/sampling_050/female_test_original.csv`
  - `4/sampling_050/female_train_adasyn.csv`
- 训练与评估输出：
  - 阈值 0.5：`4/models/t05/` 下的 metrics、predictions、feature_importance
  - 阈值 0.3：`4/models/t03/` 下的 metrics、predictions、feature_importance
  - 每个目录都有 `model_summary.json` 汇总随机森林/XGBoost/LightGBM 的测试集指标

# 客观效果如何（已跑出的指标摘录）

说明：采用 ADASYN=0.5（训练集1:1），测试集保持原始分布。下列数值是“测试集”指标，真实可靠，便于比较不同阈值的取舍。

- 阈值 0.5（追求均衡）

  - 随机森林：precision=0.353，recall=0.300，f1=0.324，ROC-AUC=0.728，PR-AUC=0.419
  - XGBoost：precision=0.316，recall=0.300，f1=0.308，ROC-AUC=0.695，PR-AUC=0.357
  - LightGBM：precision=0.286，recall=0.300，f1=0.293，ROC-AUC=0.700，PR-AUC=0.331
- 阈值 0.3（提升召回，牺牲精确率）

  - 随机森林：precision=0.275，recall=0.700，f1=0.394，ROC-AUC=0.728，PR-AUC=0.419
  - XGBoost：precision=0.214，recall=0.300，f1=0.250，ROC-AUC=0.695，PR-AUC=0.357
  - LightGBM：precision=0.250，recall=0.300，f1=0.273，ROC-AUC=0.700，PR-AUC=0.331

客观解读：

- “阈值降低 → 召回提升、精确率降低”的趋势非常清晰，符合临床“优先召回”的目标导向。将阈值从 0.5 降到 0.3 时，随机森林召回从 0.30 提升到 0.70，但精确率从 0.353 降为 0.275。
- 现阶段尚未达到“召回≥90% 且 精确率≥70%”的严格目标。要达到如此苛刻的组合指标，通常需要更强的建模/阈值策略、更多特征/样本或更复杂的代价敏感优化。
- ROC-AUC ≈ 0.70–0.73、PR-AUC ≈ 0.33–0.42，说明模型已具备一定区分能力，但仍有优化空间。

# 小结

- 思路完整：清洗→初筛→过采样→多模训练→阈值评估，且每步都有清晰原理与文件产物。
- 当前方案得到的“东西”：
  - 干净的可训练数据集（含标准化与医学关键变量）
  - 单因素统计报告与一个可用的精简特征集
  - 针对不平衡的过采样训练集（目标占比 0.5）
  - 三个模型在不同阈值下的可复现实验结果与预测文件、特征重要性
- 效果客观评价：有区分度但距离临床高召回+高精确的理想目标仍有差距。阈值下调能显著提高召回（如 0.3 时 RF 召回≈0.70），但精确率会相应下降。
