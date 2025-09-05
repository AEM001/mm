

# 关键改动
- 模型设定重构
  - 不再强制加入 l(3)+l(4)，改为“可选”的线性计数项（怀孕次数/生产次数）。
  - 系统化候选模型集合：
    - 基准平滑主效应：`s(孕周)+s(BMI)+s(年龄)`
    - 单交互：`te(孕周,BMI)`、`te(孕周,年龄)`、`te(BMI,年龄)`
    - 全两两交互：`te(孕周,BMI)+te(孕周,年龄)+te(BMI,年龄)`
    - 可选线性计数项：+怀孕次数、+生产次数、+两者，以及“主效应+线性计数项+全交互”的全项模型
- 平滑复杂度与交互复杂度可调
  - 暴露参数：
    - `n_splines_s`（单变量平滑项基函数数，默认20）
    - `n_splines_te`（交互张量项每维基函数数，默认10）
- 调参与评估升级
  - 每个候选模型先尝试 λ 网格搜索（`np.logspace(-3,3,7)`）；若 gridsearch 失败则自动退回直接拟合，稳健不崩。
  - 交叉验证升级为重复 K 折（默认 5 折×3 次），指标含 MAE、RMSE、MedAE、R²。
  - 可选保存每折预测与残差：[gam_enhanced_results/cv_predictions.csv](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_results/cv_predictions.csv:0:0-0:0)（含样本ID）。
- 诊断增强
  - 残差 vs 拟合值图 + Q-Q 图
  - Shapiro-W、Jarque-Bera 正态性检验
  - Breusch–Pagan 异方差检验（若系统有 statsmodels）
- CLI 可配置
  - 新增命令行参数：`--n_splines_s`、`--n_splines_te`、`--k`、`--repeats`、`--no_save_fold_predictions`

# 运行与结果
我已在目录 `~/Downloads/mm/1` 运行脚本并生成输出到 `gam_enhanced_results/`：
- [best_model_summary.txt](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_results/best_model_summary.txt:0:0-0:0)
- [detailed_model_summary.csv](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_results/detailed_model_summary.csv:0:0-0:0)
- [partial_dependence_plots.png](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_results/partial_dependence_plots.png:0:0-0:0)
- [model_diagnostics.png](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_results/model_diagnostics.png:0:0-0:0)
- [diagnostic_statistics.csv](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_results/diagnostic_statistics.csv:0:0-0:0)
- [cv_results.csv](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_results/cv_results.csv:0:0-0:0)
- [cv_predictions.csv](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_results/cv_predictions.csv:0:0-0:0)（每折预测+残差，含样本ID）

主要结论：
- AICc 最优模型：`M2_interact_WB`（`s(孕周)+s(BMI)+s(年龄)+te(孕周,BMI)`）
  - 伪R² ≈ 0.2322
  - AICc ≈ 1,042,287.85
- 重复 5 折 × 3 次 CV（按 RMSE_mean 排序）：
  - CV最优模型为 `M4_interact_BA`（`te(BMI,年龄)`）：
    - RMSE_mean ≈ 0.031014
    - MAE_mean ≈ 0.024698
    - R2_mean ≈ 0.1451
  - 说明：信息准则（AICc）与泛化性能（CV）在此存在偏好差异。若更重视泛化表现，建议采用 `M4_interact_BA`；若更重视整体拟合/信息准则，采用 `M2_interact_WB`。

# 如何再次运行
在 `~/Downloads/mm/1/` 目录下：
- 默认参数（保存每折预测）：
  ```
  python3 gam_enhanced_modeling.py
  ```
- 自定义平滑复杂度与 CV 设置：
  ```
  python3 gam_enhanced_modeling.py --n_splines_s 15 --n_splines_te 8 --k 10 --repeats 5
  ```
- 不保存每折预测（只要总表）：
  ```
  python3 gam_enhanced_modeling.py --no_save_fold_predictions
  ```

# 文件与函数更新点
- [1/gam_enhanced_modeling.py](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_modeling.py:0:0-0:0)
  - 新签名：[gam_enhanced_modeling(n_splines_s=20, n_splines_te=10, k=5, repeats=3, save_fold_predictions=True)](cci:1://file:///Users/Mac/Downloads/mm/1/gam_enhanced_modeling.py:31:0-193:72)
  - 新函数（或重构）：
    - [build_model_specs(feature_names, n_splines_s, n_splines_te)](cci:1://file:///Users/Mac/Downloads/mm/1/gam_enhanced_modeling.py:599:0-671:16)：定义候选模型（含可选线性计数项与交互）
    - [cross_validate_all_models(..., save_fold_predictions, output_dir, sample_ids)](cci:1://file:///Users/Mac/Downloads/mm/1/gam_enhanced_modeling.py:498:0-592:13)：重复K折CV，保存每折预测与残差
  - 诊断函数 [perform_model_diagnostics()](cci:1://file:///Users/Mac/Downloads/mm/1/gam_enhanced_modeling.py:384:0-479:80) 增加正态性与异方差检验
  - 顶部增加 `warnings` 与可选 `statsmodels` 导入，保证在缺失 `statsmodels` 时也能正常运行
  - CLI 参数解析：`argparse`

# 你可以怎么用
- 查看并对比模型：
  - [gam_enhanced_results/cv_results.csv](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_results/cv_results.csv:0:0-0:0)：各模型 MAE/RMSE/MedAE/R² 平均与标准差
  - [gam_enhanced_results/best_model_summary.txt](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_results/best_model_summary.txt:0:0-0:0)：AICc最优模型详细 summary
  - [gam_enhanced_results/detailed_model_summary.csv](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_results/detailed_model_summary.csv:0:0-0:0)：项级别统计
- 误差分析：
  - [gam_enhanced_results/cv_predictions.csv](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_results/cv_predictions.csv:0:0-0:0)：每折的预测与残差（含样本ID），可按任意维度进行聚合或诊断
  - [model_diagnostics.png](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_results/model_diagnostics.png:0:0-0:0)、[diagnostic_statistics.csv](cci:7://file:///Users/Mac/Downloads/mm/1/gam_enhanced_results/diagnostic_statistics.csv:0:0-0:0)

# 后续可选增强
- 扩大 λ 搜索范围或采用分项 λ 网格，以进一步稳健化调参。
- 进行嵌套交叉验证（外层评估、内层调参），获得无偏的泛化估计。
- 如果认为 Y 为比例数据更适合 Beta 回归或其他族，可以切换到相应的回归族（需更换库或自行实现），或对 `LinearGAM` 结果做稳健性对照。
- 输出偏残差图、逐项退化检验（drop-term tests）等，以增强解释性。

如需我把 CV 排名第一的 `M4_interact_BA` 固化为“主模型”输出（比如优先画偏依赖图与摘要），或再细化交互搜索/特征工程，请告诉我你的偏好。