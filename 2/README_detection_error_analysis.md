# NIPT检测误差分析模块

## 概述

本模块支持多种BMI分组方法的NIPT检测误差分析，包括：
1. **区间删失生存模型分组**（原有方法）
2. **BMI分界值分组**（新方法）
3. **自定义分组策略**（扩展性支持）

## 主要特性

- **策略模式设计**：支持多种分组方法，易于扩展
- **向后兼容**：保持问题三的代码正常工作
- **灵活配置**：可以单独运行或比较不同策略
- **完整分析**：包括噪声估计、误差分析、蒙特卡洛模拟等

## 使用方法

### 1. 使用原有的区间删失生存模型方法

```python
from detection_error_analysis import main

# 使用原有方法（默认）
result = main(strategy_type='interval_censored')
```

### 2. 使用新的BMI分界值方法

```python
from detection_error_analysis import main

# 使用新的BMI分界值方法
result = main(strategy_type='bmi_boundary')
```

### 3. 比较不同策略

```python
from detection_error_analysis import compare_strategies

# 比较所有可用策略
results = compare_strategies()
```

### 4. 自定义分组策略

```python
from detection_error_analysis import GroupingStrategy, run_error_analysis_with_strategy

class CustomStrategy(GroupingStrategy):
    def get_groups_and_true_times(self, data_file):
        # 实现自定义分组逻辑
        bmi_groups = ...  # DataFrame with columns: ['组别', 'BMI区间', ...]
        true_times = ...  # numpy array
        return bmi_groups, true_times
    
    def get_strategy_name(self):
        return "自定义分组策略"

# 使用自定义策略
strategy = CustomStrategy()
result = run_error_analysis_with_strategy(strategy, data_file, 'custom')
```

## 文件结构

```
2/
├── detection_error_analysis.py          # 主模块（已重构）
├── run_bmi_boundary_analysis.py         # BMI分界值方法示例
├── README_detection_error_analysis.md   # 本文档
├── detection_error_results_interval_censored/  # 区间删失方法结果
├── detection_error_results_bmi_boundary/       # BMI分界值方法结果
└── ...
```

## 输出结果

每种策略的结果保存在独立的目录中：
- `detection_error_results_interval_censored/` - 区间删失方法结果
- `detection_error_results_bmi_boundary/` - BMI分界值方法结果

每个结果目录包含：
- `detection_error_analysis/` - 误差分析子目录
  - `measurement_noise_details.csv` - 测量噪声详细数据
  - `detection_error_analysis.csv` - 检测误差分析数据
  - `monte_carlo_simulation.csv` - 蒙特卡洛模拟结果
  - `detection_error_analysis.png` - 可视化图表
  - `detection_error_analysis_report.md` - 分析报告

## 兼容性说明

- **问题三的代码无需修改**：继续使用原有的`DetectionErrorAnalyzer`类
- **API保持一致**：所有原有的函数接口都保持不变
- **扩展性良好**：可以轻松添加新的分组策略

## 依赖关系

### 区间删失生存模型策略
- `interval_censored_survival_model.py`

### BMI分界值策略  
- `nipt_timing_optimization.py`（位于目录3中）

### 问题三的复用
- 无额外依赖，直接使用原有接口

## 示例运行

```bash
# 运行BMI分界值分析
python run_bmi_boundary_analysis.py

# 或直接运行主模块
python detection_error_analysis.py
```

## 注意事项

1. **数据文件**：确保`processed_data.csv`文件存在
2. **依赖模块**：根据使用的策略确保相应的依赖模块可用
3. **路径配置**：代码会自动处理不同目录间的导入关系
4. **结果目录**：不同策略的结果保存在不同目录中，避免覆盖

## 错误处理

- 如果某个策略失败，其他策略仍可正常运行
- 详细的错误信息会输出到控制台
- 可以通过比较功能来验证不同策略的稳定性
