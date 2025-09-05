# 问题二建模方案：GAM + BMI分组的联合优化（含边界搜索）

## 一句话回答
- 可以把“BMI分组边界”纳入优化变量，与“各组最佳NIPT检测时点”一起做联合优化。推荐采用一维分段最优化：在按BMI排序的样本序列上，用动态规划（DP）做“最优分段 + 组内最优检测时点”的全局搜索；也可从“均匀分5组”的起点用局部坐标下降细化边界（等价于你提出的思路）。

---

## 背景与数据约定
- 任务目标：对男胎孕妇的 BMI 做合理分组，并给出每组的最佳 NIPT 检测时点，使总体潜在风险最小，同时分析检测误差影响。
- 已有数据与脚本：
  - `2/data_process.py`：生成问题二数据集，每位孕妇取“首次 Y 浓度≥4%”的记录，导出 `2/processed_data_problem2.csv`，包含 `BMI` 与 `最早达标孕周`。
  - `2/clustering_analysis.py`：以 BMI 一维 K-means 分组，按 10%分位数（带 12 周逻辑）给出各组“最佳时点”；输出 `2/results/optimal_timing_summary.csv`。当前结果示例：3 组，时点约 11.37、11.7、12.26 周（覆盖率设为 90%）。
  - `2/error_analysis.py`：对 Y 浓度与 BMI 的 ±5% 测量误差做 Monte Carlo 模拟，评估“最佳时点”的稳定性。
- 第一问（GAM）回顾：
  - 训练代码见 `1/main.py`，模型规格见 `1/model_specs.py`。
  - AICc 最优模型为 `M2_interact_WB`（周龄×BMI 张量交互，Pseudo R²≈0.2322，详见 `1/gam_enhanced_results/best_model_summary.txt`）。
  - 重复K折CV上，误差指标之间差异不大（`cv_results.csv`），`M4_interact_BA` 的 R²≈0.145 最好，`M2` 的 R²≈0.139。综合选择 `M2` 作为“时点/阈值”相关的业务解读主力（更偏 AICc 与可解释的周龄×BMI 交互）。

---

## 关键建模思想
- 我们把“BMI 分组边界”和“各组最佳检测时点 T_g”作为联合优化变量；目标函数显式刻画两类风险：
  1) 检测过早（在个体达到 Y≥4% 前抽血）导致的失败/复测风险；
  2) 检测过晚（尤其超过 12/13/28 周阈值）导致的临床窗口缩短风险。
- 核心在于一维有序变量（BMI）上的分段最优化，可用 DP 在多项式时间求全局最优；你提出的“均匀5组起点 + 边界搜索”对应局部坐标下降，是一种高效近似方案，可作为 DP 的快速替代或热启动。

---

## 变量与定义
- 数据 D = { (bmi_i, w_i) }，i=1..n：
  - bmi_i：第 i 位孕妇的 BMI。
  - w_i：该孕妇“最早达标孕周”（Y 浓度首次≥4%）。可用两种方式获得：
    - 观测法：直接用 `2/data_process.py` 的 `最早达标孕周`；
    - 预测法（推荐用于稳健性校准）：用第一问 GAM（`M2_interact_WB`），在 t ∈ [10,25] 周网格上，预测 ŷ(t, bmi_i, age=均值, 其他=均值)，取最早 ŷ≥0.04 的 t 作为 ŵ_i。
- 分组结构：按 BMI 升序划分为 K 个不重叠区间 G_1, ..., G_K；组 g 的检测时点为 T_g。
- 覆盖率：cov_g(T) = P(w ≤ T | BMI ∈ G_g)，经验估计为组内比例。

---

## 风险函数与目标
- 设定目标：在满足覆盖率约束的前提下，使“检测时点过晚”的总体风险最小。
- 覆盖率约束（建议）：cov_g(T_g) ≥ c（例如 c=0.90）。在经验上，这让 T_g 至少为组内 w 的 c 分位数。
- 过晚检测风险（参考题面“早期≤12周风险较低；13-27周风险高；≥28周风险极高”）：
  - 令 timeRisk(T) = 0, 若 T ≤ 12；= r1, 若 12 < T ≤ 27；= r2, 若 T > 27（可设 r1=1, r2=3）。
- 组内时点偏迟的“超额等待”惩罚（可选，细化目标）：E[(T_g - w)_+ | BMI ∈ G_g]，鼓励更靠近“最早可检测”的时点。
- 综上，整体目标可写为（加权按组样本数）：
  J = Σ_g [ N_g · timeRisk(T_g) + λ · N_g · E[(T_g - w)_+] ]
  s.t. cov_g(T_g) ≥ c，N_g ≥ N_min，K_min ≤ K ≤ K_max。
  - λ 控制“延迟检测”权重；N_min（如 ≥20）避免极小样本组；K 的选择可用信息准则或外部设定。

---

## 组内最优检测时点（给定组）
- 若采用“覆盖率约束”并忽略 E[(T-w)_+]，那么组内最优 T_g* = max{ q_c(w | G_g), 12 }（即满足90%覆盖且不早于12周的最早时点）。
- 若加入“等待惩罚”项，可在候选集合 {12, q_c, q_{c+δ}, ..., 25} 上扫描，取使 J_g 最小的 T。
- 这样，给定一个候选分组 G_1..G_K，所有 T_g* 都能在 O(n) 时间内（分位数与有限候选扫描）求得。

---

## 算法一：动态规划（DP）最优分段（推荐）
- 适用场景：一维有序变量（BMI）上做分段，组内代价仅依赖该段的数据，可全局最优。
- 预处理：按 BMI 升序重排样本，得到序列 (b_1,w_1),...,(b_n,w_n)。
- 代价定义：对任意区间 [i..j]，定义 Cost(i,j) 为该段的最小组内代价：
  - 先用该段的 w_i..w_j 计算 T*（按上节规则），
  - 再代入 J_g（含 timeRisk 与等待惩罚）得到该段代价。
- DP 递推（给定 K）：
  - DP[g][j] = min over i ∈ [g-1+N_min .. j-N_min+1] of { DP[g-1][i-1] + Cost(i,j) }。
  - 初值 DP[1][j] = Cost(1,j)。
  - 复杂度 O(K · n^2)。n≈247 时，K≤6 完全可行。
- K 的选择：
  - 方案A（外给）：如 K ∈ {2..6} 枚举，取 J 最小者；
  - 方案B（带复杂度惩罚）：在目标里加 π·K（π 为组数惩罚），单次 DP 求解。
- 输出：最优边界（通过回溯），各组区间与 T_g*，以及分组代价贡献与覆盖率。

伪代码：
```
Input: sorted w[1..n] by BMI, coverage c, penalty λ, r1,r2, N_min, K_range
Precompute: For all i<=j with (j-i+1)>=N_min:
  - compute candidate T* for segment S=i..j
  - Cost[i][j] = N_S * timeRisk(T*) + λ * sum_{t in S} max(0, T* - w_t)
Best = +∞; BestSol = None
for K in K_range:
  DP = +∞ matrix of size (K+1) x (n+1)
  Prev = argmin backpointers
  for j in 1..n:
    if j>=N_min: DP[1][j] = Cost[1][j], Prev[1][j] = 0
  for g in 2..K:
    for j in 1..n:
      for i in (g-1)*N_min .. j-N_min+1:
        cand = DP[g-1][i-1] + Cost[i][j]
        if cand < DP[g][j]: DP[g][j] = cand; Prev[g][j] = i-1
  if DP[K][n] < Best: Best = DP[K][n]; BestSol = (K, Prev)
Return BestSol with backtracking boundaries and T*_per_segment
```

---

## 算法二：局部坐标下降搜索（从“均匀分5组”起步）
- 适用场景：快速求一个强基线；与 DP 结果对比验证。
- 步骤：
  1) 按 BMI 等分为 5 组（或用 `clustering_analysis.py` 的 K-means 结果作热启动）；
  2) 固定组数 K，每次挑一个边界，向左/右移动若干个样本（保持两侧 ≥N_min），重新计算相邻两段的 T* 与代价，只要总代价下降就接受；
  3) 轮流更新各边界，直至所有单边界移动都无法继续下降（近似局部最优）。
- 复杂度：每次只重算相邻两段，接近 O(n) 每步，收敛很快。

伪代码：
```
Initialize boundaries B from equal-size K=5 (or K-means)
repeat
  improved = False
  for each boundary b in B:
    for step in {−Δ,..,−1, +1,..,+Δ}:
      move b by step samples if both adjacent groups ≥ N_min
      recompute Cost for the two affected segments with their T*
      if total_cost decreases by > ε: accept move; improved = True; break
until not improved
Return boundaries and T* per group
```

---

## 与第一问 GAM 的衔接方式（用于 w 的稳健估计与时点校准）
- 目的：观测的 `最早达标孕周` 受抽样与测量误差影响，用 GAM 进行“曲线外插/插值”可稳健估计“首次达 4% 的周龄”。
- 实施：
  1) 读取 `1/gam_enhanced_results/best_model_summary.txt` 确认最优为 `M2_interact_WB`；
  2) 载入训练均值/标准差（来自 `1/processed_data.csv`）以对 BMI、周龄、年龄做同分布标准化；
  3) 对每个个体 i ，在 t ∈ {10,10.1,..,25} 上，构造特征：`孕周_标准化`(t)、`BMI_标准化`(bmi_i)、`年龄_标准化`=0（或用样本均值），其他特征取 0；
  4) 用 GAM 预测 ŷ(t)，取最早使 ŷ(t)≥0.04 的 t 作为 ŵ_i（若始终<0.04 可标记为缺失）；
  5) 用 ŵ_i 替换或与观测 w_i 做折中（如加权平均），进入上面的 DP/坐标下降优化。
- 备注：GAM 对“周龄”的单调性未强约束，个别 BMI 区域可能出现非单调；可对 ŷ(t) 做移动平均或单调回归（Pool-Adjacent-Violators）再找阈值，确保医学可解释性。

---

## 与现有代码的落地对接
- 保留现有基线：`2/clustering_analysis.py`（K-means 一维分组 + 分位数时点），用于结果对照与热启动。
- 新增脚本建议：`2/segmentation_optimization.py`
  - 功能：
    - 读取 `2/processed_data_problem2.csv` 与（可选）`1/gam_enhanced_results/` 下最优 GAM；
    - 提供两路优化器：DP（全局）与坐标下降（局部/热启动）；
    - 超参：K 或 K_range、c（覆盖率，默认 0.9）、N_min（默认 20）、λ、r1、r2、Δ、ε；
    - 输出：`2/results/segment_optimized_summary.csv`（各组 BMI 区间、T*、覆盖率、样本数、风险），及边界可视化图。
  - 关键函数：
    - `estimate_w_from_gam(bmi)`：用 GAM + 网格搜索求 ŵ；
    - `segment_cost(i,j, params)`：计算区间 [i..j] 的 T* 与代价；
    - `dp_optimize(K or K_range, params)`：DP 求最优分段；
    - `coordinate_descent(init_boundaries, params)`：从初始边界做局部细化。

---

## 误差与稳健性分析（与 `2/error_analysis.py` 一致/增强）
- Monte Carlo：对 `Y染色体浓度` 与 `BMI` 注入 ±5% 误差，重新生成 `w` 或 `ŵ`，重复运行优化，统计 T* 的分布与边界位置的波动；
- Bootstrap：对样本做有放回重采样，重复 DP/坐标下降，得到各边界与 T* 的置信区间（如 95%）；
- 敏感性：对 c（覆盖率阈值）、N_min、λ、r1、r2 做网格敏感性分析，绘制 T* 与总体风险的响应曲线。

---

## 实验与报告模板
- 报告条目：
  1) 分组与时点方案表（区间、样本数、覆盖率、T*、10/20/25%分位）；
  2) 与 K-means 基线的对比（总体风险 J、各组风险、总覆盖率差异）；
  3) 误差/Bootstrap 稳健性结果（均值±标准差与区间）；
  4) 可视化：BMI-最早达标孕周散点 + 分段边界与 T* 水平线；
  5) 结论与建议（如某些高 BMI 组建议预约至 ≥12 周，低 BMI 组可提前至 11.3 周等）。

---

## 推荐默认参数
- 覆盖率阈值 c = 0.90；
- 组数 K ∈ {2,3,4,5,6}（枚举 + DP 选择）或固定 K=3 与基线对比；
- 最小组样本数 N_min = 20（与当前 `optimal_timing_summary.csv` 第三组 n=20 一致）；
- 时间风险权重 r1=1, r2=3；延迟惩罚 λ 可从 0~1 网格搜索；
- 局部搜索步长 Δ=3（样本数），改进阈值 ε=1e-4。

---

## 小结
- 你的想法“把分组一起纳入最优化并从均匀5组出发搜索”完全可行；
- 在一维 BMI 上使用 DP 能保证全局最优，计算量也可控；局部坐标下降可提供快速强基线或作为 DP 的热启动；
- 引入第一问的 GAM 可稳健估计“最早达标孕周”，提升时点选择的抗噪性与解释力；
- 全流程与现有脚本天然衔接，产出表格与图形可直接放入竞赛论文。
