#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题三（GAMM+KMeans）检测误差影响分析
- 基于 `3/gamm_y_chromosome_prediction.py` 的 R/mgcv GAMM via rpy2 预测
- 基于 `3/kmeans_bmi_segmentation.py` 的 KMeans 分群
- 问题三专属误差分析模块（本文件内独立实现，不复用问题二）

输出目录：`3/gamm_detection_error_analysis/`
    └── detection_error_analysis/
        ├── measurement_noise_details.csv
        ├── detection_error_analysis.csv
        ├── monte_carlo_simulation.csv
        ├── detection_error_analysis.png
        └── detection_error_analysis_report.md
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# 项目根路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 设置中文字体（跨设备一致渲染）
try:
    from set_chinese_font import set_chinese_font
    set_chinese_font()
except Exception:
    pass

# 导入问题三的分析器（GAMM + KMeans）
from kmeans_bmi_segmentation import BMISegmentationAnalyzer


class Q3DetectionErrorAnalyzer:
    """
    问题三专属：NIPT检测误差分析器（独立实现）
    - 负责：测量噪声估计、误判率分析、蒙特卡洛模拟、图形与报告
    - 注意：阈值使用logit(4%)，数据列名沿用问题三的原始逐次数据
    """

    def __init__(self, output_dir='.', threshold_logit=-3.178054):
        self.output_dir = output_dir
        self.error_dir = os.path.join(output_dir, 'detection_error_analysis')
        os.makedirs(self.error_dir, exist_ok=True)

        self.threshold_logit = threshold_logit
        self.threshold_percentage = 0.04

        # 保存分析上下文与结果
        self.report_context = {}
        self.noise_metrics = {}
        self.error_rates = {}
        self.simulation_results = {}

    def estimate_measurement_noise(self, df, col_woman='孕妇代码', col_y='Y染色体浓度',
                                   col_week='孕周_标准化', smoothing_window=3):
        print("=== 测量噪声估计（Q3） ===")

        noise_data = []
        for woman_code, woman_data in df.groupby(col_woman):
            if len(woman_data) < 3:
                continue

            if col_week and col_week in woman_data.columns:
                woman_data = woman_data.sort_values(col_week).reset_index(drop=True)
            else:
                woman_data = woman_data.reset_index(drop=True)

            y_values = woman_data[col_y].values

            # 方法1: 高斯平滑去噪
            if len(y_values) >= smoothing_window:
                y_smooth = gaussian_filter1d(y_values, sigma=1.0)
                residuals_smooth = y_values - y_smooth
                noise_smooth = np.std(residuals_smooth)
            else:
                noise_smooth = np.nan

            # 方法2: 相邻差分（MAD）
            if len(y_values) >= 2:
                diffs = np.diff(y_values)
                noise_diff = np.median(np.abs(diffs - np.median(diffs))) / 0.6745
            else:
                noise_diff = np.nan

            # 方法3: 线性拟合残差
            if len(y_values) >= 3:
                x_idx = np.arange(len(y_values))
                try:
                    coeffs = np.polyfit(x_idx, y_values, 1)
                    y_fit = np.polyval(coeffs, x_idx)
                    residuals_linear = y_values - y_fit
                    noise_linear = np.std(residuals_linear)
                except Exception:
                    noise_linear = np.nan
            else:
                noise_linear = np.nan

            signal_std = np.std(y_values)

            noise_data.append({
                '孕妇代码': woman_code,
                '测量次数': len(y_values),
                '噪声_平滑': noise_smooth,
                '噪声_差分': noise_diff,
                '噪声_线性': noise_linear,
                'SNR_平滑': signal_std / noise_smooth if noise_smooth and not np.isnan(noise_smooth) and noise_smooth > 0 else np.nan,
                'SNR_差分': signal_std / noise_diff if noise_diff and not np.isnan(noise_diff) and noise_diff > 0 else np.nan,
                'SNR_线性': signal_std / noise_linear if noise_linear and not np.isnan(noise_linear) and noise_linear > 0 else np.nan,
            })

        noise_df = pd.DataFrame(noise_data)
        noise_summary = {
            '样本数': len(noise_df),
            '平均测量次数': noise_df['测量次数'].mean() if len(noise_df) else 0,
            '噪声水平': {
                '平滑法': {
                    '均值': noise_df['噪声_平滑'].mean(),
                    '中位数': noise_df['噪声_平滑'].median(),
                    '标准差': noise_df['噪声_平滑'].std(),
                },
                '差分法': {
                    '均值': noise_df['噪声_差分'].mean(),
                    '中位数': noise_df['噪声_差分'].median(),
                    '标准差': noise_df['噪声_差分'].std(),
                },
                '线性法': {
                    '均值': noise_df['噪声_线性'].mean(),
                    '中位数': noise_df['噪声_线性'].median(),
                    '标准差': noise_df['噪声_线性'].std(),
                },
            },
            '信噪比': {
                'SNR_平滑': noise_df['SNR_平滑'].mean(),
                'SNR_差分': noise_df['SNR_差分'].mean(),
                'SNR_线性': noise_df['SNR_线性'].mean(),
            },
        }

        noise_df.to_csv(os.path.join(self.error_dir, 'measurement_noise_details.csv'), index=False, encoding='utf-8')
        self.noise_metrics = noise_summary

        print(f"噪声估计完成，共分析 {len(noise_df)} 个样本")
        return noise_summary, noise_df

    def analyze_detection_errors(self, df, col_woman='孕妇代码', col_y='Y染色体浓度', noise_level=None):
        print("=== 检测误差分析（Q3） ===")

        if noise_level is None:
            if self.noise_metrics:
                noise_level = self.noise_metrics['噪声水平']['平滑法']['均值']
            else:
                noise_level = 0.1

        y_range = np.linspace(-5, 0, 100)  # logit范围
        rows = []
        for true_conc in y_range:
            if true_conc < self.threshold_logit:
                fp = 1 - stats.norm.cdf((self.threshold_logit - true_conc) / noise_level)
            else:
                fp = 0.0

            if true_conc >= self.threshold_logit:
                fn = stats.norm.cdf((self.threshold_logit - true_conc) / noise_level)
            else:
                fn = 0.0

            rows.append({
                '真实浓度_logit': true_conc,
                '真实浓度_百分比': 1 / (1 + np.exp(-true_conc)),
                '假阳性概率': fp,
                '假阴性概率': fn,
                '正确检测概率': 1 - fp - fn,
                '距离阈值': abs(true_conc - self.threshold_logit),
            })

        error_df = pd.DataFrame(rows)

        near_th = error_df[error_df['距离阈值'] <= 2 * noise_level]
        error_summary = {
            '噪声水平': noise_level,
            '阈值附近误差': {
                '范围': f"±{2*noise_level:.4f} (logit单位)",
                '平均假阳性率': near_th['假阳性概率'].mean(),
                '平均假阴性率': near_th['假阴性概率'].mean(),
                '最大假阳性率': near_th['假阳性概率'].max(),
                '最大假阴性率': near_th['假阴性概率'].max(),
            },
            '4%阈值处误差': {
                '假阳性率': error_df.loc[np.argmin(np.abs(error_df['真实浓度_logit'] - self.threshold_logit)), '假阳性概率'],
                '假阴性率': error_df.loc[np.argmin(np.abs(error_df['真实浓度_logit'] - self.threshold_logit)), '假阴性概率'],
            },
        }

        error_df.to_csv(os.path.join(self.error_dir, 'detection_error_analysis.csv'), index=False, encoding='utf-8')
        self.error_rates = error_summary
        return error_summary, error_df

    def monte_carlo_simulation(self, true_times, bmi_groups, n_simulations=1000, noise_level=None):
        print("=== 蒙特卡洛模拟（Q3） ===")

        if noise_level is None:
            noise_level = self.noise_metrics['噪声水平']['平滑法']['均值'] if self.noise_metrics else 0.1

        results = []
        start_week, max_week = 10, 30

        for i, (_, group) in enumerate(bmi_groups.iterrows()):
            true_time = true_times[i] if i < len(true_times) else 15.0
            detected_times = []

            for _ in range(n_simulations):
                weeks = np.arange(start_week, max_week + 1)

                if true_time <= start_week:
                    true_conc = np.full(len(weeks), self.threshold_logit + 0.5)
                else:
                    slope = (self.threshold_logit - (self.threshold_logit - 1)) / (true_time - start_week)
                    true_conc = (self.threshold_logit - 1) + slope * (weeks - start_week)
                    true_conc = np.maximum(true_conc, self.threshold_logit - 1)

                measured = true_conc + np.random.normal(0, noise_level, len(weeks))
                qualified_idx = np.where(measured >= self.threshold_logit)[0]
                dt = weeks[qualified_idx[0]] if len(qualified_idx) > 0 else max_week
                detected_times.append(dt)

            detected_times = np.array(detected_times)
            results.append({
                '组别': group['组别'],
                'BMI区间': group['BMI区间'],
                '真实达标时间': true_time,
                '预测均值': float(np.mean(detected_times)),
                '预测中位数': float(np.median(detected_times)),
                '偏差': float(np.mean(detected_times) - true_time),
                'RMSE': float(np.sqrt(np.mean((detected_times - true_time) ** 2))),
                'MAE': float(np.mean(np.abs(detected_times - true_time))),
                '标准差': float(np.std(detected_times)),
                'CI_2.5%': float(np.percentile(detected_times, 2.5)),
                'CI_97.5%': float(np.percentile(detected_times, 97.5)),
                '检测成功率': float(np.mean(detected_times < max_week)),
            })

        sim_df = pd.DataFrame(results)
        sim_df.to_csv(os.path.join(self.error_dir, 'monte_carlo_simulation.csv'), index=False, encoding='utf-8')

        self.simulation_results = {
            '总体偏差': {
                '平均偏差': sim_df['偏差'].mean(),
                '平均RMSE': sim_df['RMSE'].mean(),
                '平均MAE': sim_df['MAE'].mean(),
            },
            '各组结果': sim_df.to_dict('records'),
        }
        return self.simulation_results, sim_df


def create_error_visualizations(analyzer, noise_df, error_df, sim_df):
    print("=== 创建误差分析可视化（Q3） ===")
    # 尽量统一字体渲染
    try:
        from set_chinese_font import set_chinese_font
        set_chinese_font()
    except Exception:
        pass

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 噪声分布
    ax1 = axes[0, 0]
    ax1.boxplot([noise_df['噪声_平滑'].dropna(), noise_df['噪声_差分'].dropna(), noise_df['噪声_线性'].dropna()],
                labels=['平滑法', '差分法', '线性法'])
    ax1.set_ylabel('噪声水平')
    ax1.set_title('不同方法估计的噪声水平分布')
    ax1.grid(True, alpha=0.3)

    # 2. SNR分布
    ax2 = axes[0, 1]
    ax2.boxplot([noise_df['SNR_平滑'].dropna(), noise_df['SNR_差分'].dropna(), noise_df['SNR_线性'].dropna()],
                labels=['平滑法', '差分法', '线性法'])
    ax2.set_ylabel('信噪比')
    ax2.set_title('不同方法计算的信噪比分布')
    ax2.grid(True, alpha=0.3)

    # 3. 假阳性/假阴性曲线
    ax3 = axes[0, 2]
    conc_percent = error_df['真实浓度_百分比'] * 100
    ax3.plot(conc_percent, error_df['假阳性概率'], 'r-', label='假阳性率', linewidth=2)
    ax3.plot(conc_percent, error_df['假阴性概率'], 'b-', label='假阴性率', linewidth=2)
    ax3.axvline(x=4, color='black', linestyle='--', alpha=0.7, label='4%阈值')
    ax3.set_xlabel('真实Y染色体浓度 (%)')
    ax3.set_ylabel('误判概率')
    ax3.set_title('检测误差概率曲线')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 10)

    # 4. 偏差条形图
    ax4 = axes[1, 0]
    groups = [f"第{row['组别']}组" for _, row in sim_df.iterrows()]
    biases = sim_df['偏差'].values
    colors = ['red' if b > 0 else 'blue' for b in biases]
    bars = ax4.bar(groups, biases, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_ylabel('时间偏差 (周)')
    ax4.set_title('各BMI组达标时间估计偏差')
    ax4.grid(True, alpha=0.3)
    for bar, bias in zip(bars, biases):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.1 if height > 0 else height - 0.1,
                 f'{bias:.1f}', ha='center', va='bottom' if height > 0 else 'top')

    # 5. RMSE与MAE
    ax5 = axes[1, 1]
    x_pos = np.arange(len(groups))
    width = 0.35
    ax5.bar(x_pos - width/2, sim_df['RMSE'], width, label='RMSE', alpha=0.7)
    ax5.bar(x_pos + width/2, sim_df['MAE'], width, label='MAE', alpha=0.7)
    ax5.set_xlabel('BMI组')
    ax5.set_ylabel('误差 (周)')
    ax5.set_title('各组预测误差对比')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(groups)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 置信区间
    ax6 = axes[1, 2]
    pred_means = sim_df['预测均值']
    ci_lower = sim_df['CI_2.5%']
    ci_upper = sim_df['CI_97.5%']
    for i, g in enumerate(groups):
        lower_err = max(0, pred_means.iloc[i] - ci_lower.iloc[i])
        upper_err = max(0, ci_upper.iloc[i] - pred_means.iloc[i])
        ax6.errorbar(i, pred_means.iloc[i], yerr=[[lower_err], [upper_err]], fmt='o', capsize=5,
                     label='预测' if i == 0 else "")
    ax6.set_xlabel('BMI组')
    ax6.set_ylabel('达标时间 (周)')
    ax6.set_title('达标时间预测置信区间')
    ax6.set_xticks(range(len(groups)))
    ax6.set_xticklabels(groups)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(analyzer.error_dir, 'detection_error_analysis.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"误差分析图表已保存为: {out}")


def generate_error_report(analyzer):
    print("=== 生成误差分析报告（Q3） ===")

    analysis_basis = analyzer.report_context.get('analysis_basis', 'GAMM+KMeans聚类分组')
    report_title = analyzer.report_context.get('report_title', '问题三（GAMM+KMeans）检测误差分析报告')
    report = f"""# {report_title}

## 1. 分析概述

本报告分析NIPT检测过程中测量误差及其对达标时间估计的影响，基于{analysis_basis}的结果进行误差传播分析。
"""

    group_summary = analyzer.report_context.get('group_summary')
    if group_summary:
        report += f"""

## 1.1 分组设定与依据

{group_summary}
"""

    report += """
## 2. 测量噪声分析

### 2.1 噪声水平估计
"""

    if analyzer.noise_metrics:
        noise = analyzer.noise_metrics
        report += f"""
- 样本数量: {noise['样本数']}个孕妇
- 平均测量次数: {noise['平均测量次数']:.1f}次

#### 不同估计方法的噪声水平：
- 平滑法: 均值={noise['噪声水平']['平滑法']['均值']:.6f}, 中位数={noise['噪声水平']['平滑法']['中位数']:.6f}
- 差分法: 均值={noise['噪声水平']['差分法']['均值']:.6f}, 中位数={noise['噪声水平']['差分法']['中位数']:.6f}
- 线性法: 均值={noise['噪声水平']['线性法']['均值']:.6f}, 中位数={noise['噪声水平']['线性法']['中位数']:.6f}

#### 信噪比分析：
- SNR (平滑法): {noise['信噪比']['SNR_平滑']:.2f}
- SNR (差分法): {noise['信噪比']['SNR_差分']:.2f}
- SNR (线性法): {noise['信噪比']['SNR_线性']:.2f}
"""

    report += """
## 3. 检测误差分析

### 3.1 假阳性/假阴性概率
"""

    if analyzer.error_rates:
        error = analyzer.error_rates
        report += f"""
- 使用噪声水平: {error['噪声水平']:.6f}

#### 4%阈值处的误差：
- 假阳性率: {error['4%阈值处误差']['假阳性率']:.4f} ({error['4%阈值处误差']['假阳性率']*100:.2f}%)
- 假阴性率: {error['4%阈值处误差']['假阴性率']:.4f} ({error['4%阈值处误差']['假阴性率']*100:.2f}%)

#### 阈值附近区域的误差：
- 范围: {error['阈值附近误差']['范围']}
- 平均假阳性率: {error['阈值附近误差']['平均假阳性率']:.4f} ({error['阈值附近误差']['平均假阳性率']*100:.2f}%)
- 平均假阴性率: {error['阈值附近误差']['平均假阴性率']:.4f} ({error['阈值附近误差']['平均假阴性率']*100:.2f}%)
- 最大假阳性率: {error['阈值附近误差']['最大假阳性率']:.4f} ({error['阈值附近误差']['最大假阳性率']*100:.2f}%)
- 最大假阴性率: {error['阈值附近误差']['最大假阴性率']:.4f} ({error['阈值附近误差']['最大假阴性率']*100:.2f}%)
"""

    report += """
## 4. 蒙特卡洛模拟结果

### 4.1 达标时间估计偏差
"""

    if analyzer.simulation_results:
        sim = analyzer.simulation_results
        report += f"""
#### 总体误差指标：
- 平均时间偏差: {sim['总体偏差']['平均偏差']:.2f}周
- 平均RMSE: {sim['总体偏差']['平均RMSE']:.2f}周
- 平均MAE: {sim['总体偏差']['平均MAE']:.2f}周

#### 各BMI组详细结果：
"""

        for group_result in sim['各组结果']:
            report += f"""
##### {group_result['BMI区间']}
- 真实达标时间: {group_result['真实达标时间']:.1f}周
- 预测均值: {group_result['预测均值']:.1f}周
- 时间偏差: {group_result['偏差']:+.1f}周
- RMSE: {group_result['RMSE']:.1f}周
- 95%置信区间: [{group_result['CI_2.5%']:.1f}, {group_result['CI_97.5%']:.1f}]周
- 检测成功率: {group_result['检测成功率']*100:.1f}%
"""

    report += f"""
## 5. 结论与建议

### 5.1 主要发现
1. 噪声水平：多方法估计结果一致性较好
2. 误判风险：4%阈值附近的假阳性/假阴性风险需在策略中考虑
3. 时间偏差：误差将引入系统性偏差，对接近阈值的情况影响更明显

### 5.2 建议
1. 重复检测：接近阈值的样本建议复检
2. 动态监测：多时点检测以提高准确性
3. 个体化阈值：结合BMI与临床特征调整阈值

### 5.3 方法改进方向
1. 噪声建模：引入时间相关噪声
2. 贝叶斯方法：整合先验信息
3. 更丰富的GAMM结构：优化平滑项与随机效应

## 6. 技术说明

- 分析基础: {analysis_basis}
- 噪声估计: 多方法交叉验证
- 误差分析: 基于正态噪声假设
- 模拟验证: 蒙特卡洛方法

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    report_path = os.path.join(analyzer.error_dir, 'detection_error_analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"误差分析报告已保存为: {report_path}")
    return report


def build_cluster_groups(cluster_df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """
    基于 KMeans 聚类结果构造 BMI 分组信息（用于误差模拟的标签与展示）。
    返回列：['组别', 'BMI区间']
    说明：这里的 BMI 区间为标准化范围的字符串描述，仅用于标签展示。
    """
    groups = []
    for cid in sorted(cluster_df['聚类标签'].unique()):
        data = cluster_df[cluster_df['聚类标签'] == cid]
        bmi_min = float(data['BMI_标准化'].min()) if len(data) else np.nan
        bmi_max = float(data['BMI_标准化'].max()) if len(data) else np.nan
        groups.append({
            '组别': int(cid + 1),
            'BMI区间': f'BMI(标准化)∈[{bmi_min:.3f}, {bmi_max:.3f}]'
        })
    groups_df = pd.DataFrame(groups).sort_values('组别').reset_index(drop=True)
    # 若传入的 n_clusters 与实际聚类标签不一致，按实际为准
    return groups_df


def extract_true_times_by_cluster(cluster_df: pd.DataFrame) -> np.ndarray:
    """
    为每个聚类生成“真实达标时间”（用于误差模拟的基准），
    采用各簇内“预测达标孕周”的中位数（从标准化还原到原始周）。
    """
    # 与 kmeans_bmi_segmentation.py 保持一致的去标准化参数
    time_mean = 16.846
    time_std = 4.076

    true_times = []
    for cid in sorted(cluster_df['聚类标签'].unique()):
        data = cluster_df[cluster_df['聚类标签'] == cid]
        if len(data) == 0:
            true_times.append(np.nan)
            continue
        median_std = float(data['预测达标孕周'].median())
        median_original = median_std * time_std + time_mean
        true_times.append(median_original)
    return np.array(true_times)


def main():
    print("=== 问题三：GAMM+KMeans 检测误差影响分析 开始 ===")

    # 1) 路径与输出目录
    data_file = os.path.join(SCRIPT_DIR, 'processed_data.csv')
    output_dir = os.path.join(SCRIPT_DIR, 'gamm_detection_error_analysis')
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {os.path.relpath(output_dir, SCRIPT_DIR)}")

    # 2) 初始化 (GAMM + KMeans) 分析器
    bmi_analyzer = BMISegmentationAnalyzer(output_dir=output_dir + os.sep)

    # 检查 R/mgcv 可用性：本模块严格依赖 R/mgcv GAMM（不允许Python替代）
    try:
        if not getattr(bmi_analyzer.gamm_predictor, 'use_r_gamm', False):
            raise RuntimeError("R/mgcv GAMM 不可用。本分析严格依赖 R GAMM，请安装并配置 rpy2 与 R 包 mgcv。")
    except AttributeError:
        raise RuntimeError("无法确认 R/mgcv GAMM 可用性。本分析严格依赖 R GAMM，请安装并配置 rpy2 与 R 包 mgcv。")

    # 3) 加载与准备数据（原始逐次检测数据 + 提取达标孕周目标）
    print("\n=== 加载与准备数据 ===")
    raw_df, target_df, X, y = bmi_analyzer.load_and_prepare_data(data_file)

    # 4) 训练 GAMM 并生成个体预测
    prediction_df = bmi_analyzer.train_gamm_and_predict(X, y, target_df)

    # 5) KMeans 分群（若不设置 k，将自动推荐）
    cluster_df = bmi_analyzer.perform_clustering(prediction_df, optimal_k=None)

    # 6) 构造分组信息与每组“真实达标时间”（用于误差模拟）
    groups_df = build_cluster_groups(cluster_df, n_clusters=len(cluster_df['聚类标签'].unique()))
    true_times = extract_true_times_by_cluster(cluster_df)

    print("\n=== 分组与真实时间基准 ===")
    for i, row in groups_df.iterrows():
        tt = true_times[i] if i < len(true_times) else np.nan
        print(f"组别{row['组别']}: {row['BMI区间']} | 中位达标时间(原始): {tt:.2f}周")

    # 7) 检测误差分析（噪声估计、误判率分析、蒙特卡洛模拟）
    analyzer = Q3DetectionErrorAnalyzer(output_dir=output_dir)
    # 设置问题三的报告上下文
    analyzer.report_context['analysis_basis'] = 'GAMM+KMeans聚类分组'
    analyzer.report_context['report_title'] = '问题三（GAMM+KMeans）检测误差分析报告'
    # 注入分组摘要，便于报告呈现“分组设定与依据”
    group_lines = [f"- 组别{int(r['组别'])}: {r['BMI区间']} | 中位达标时间≈{true_times[i]:.1f}周" for i, r in groups_df.iterrows()]
    analyzer.report_context['group_summary'] = "\n".join(group_lines)

    # 7.1 噪声估计（基于逐次检测：Y染色体浓度为logit单位，阈值为logit(4%)）
    noise_summary, noise_df = analyzer.estimate_measurement_noise(
        df=raw_df,
        col_woman='孕妇代码',
        col_y='Y染色体浓度',
        col_week='孕周_标准化',
        smoothing_window=3,
    )

    # 7.2 误判率分析（假阳性/假阴性）
    error_summary, error_df = analyzer.analyze_detection_errors(
        df=raw_df,
        col_woman='孕妇代码',
        col_y='Y染色体浓度',
        noise_level=noise_summary['噪声水平']['平滑法']['均值'] if '噪声水平' in noise_summary else None,
    )

    # 7.3 蒙特卡洛模拟（按聚类组）
    sim_summary, sim_df = analyzer.monte_carlo_simulation(
        true_times=true_times,
        bmi_groups=groups_df.rename(columns={'组别': '组别', 'BMI区间': 'BMI区间'}),
        n_simulations=1000,
        noise_level=noise_summary['噪声水平']['平滑法']['均值'] if '噪声水平' in noise_summary else None,
    )

    # 8) 可视化与报告
    create_error_visualizations(analyzer, noise_df, error_df, sim_df)
    generate_error_report(analyzer)

    print("\n=== 检测误差影响分析 完成 ===")
    print("结果目录:", os.path.relpath(analyzer.error_dir, SCRIPT_DIR))


if __name__ == '__main__':
    main()
