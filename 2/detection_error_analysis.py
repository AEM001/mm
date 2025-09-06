#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIPT检测误差分析模块
问题2：支持多种分组方法的检测误差分析

实现功能：
1. 测量噪声估计与平滑
2. 假阳性/假阴性概率分析
3. 蒙特卡洛模拟达标时间偏差
4. 误差分析可视化与报告

支持的分组方法：
- 区间删失生存模型分组
- BMI分界值分组
- 其他自定义分组方法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc
import os
import warnings
warnings.filterwarnings('ignore')

# 导入中文字体设置
import platform

def set_chinese_font():
    """设置matplotlib中文字体显示"""
    system = platform.system()
    if system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['STSong', 'Songti SC', 'STHeiti']
    elif system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'STSong']  # 其他系统尝试通用字体
    plt.rcParams['axes.unicode_minus'] = False

# 项目路径与中文字体（跨设备）
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from set_chinese_font import set_chinese_font
set_chinese_font()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


class GroupingStrategy:
    """
    分组策略基类，用于支持不同的BMI分组方法
    """
    
    def get_groups_and_true_times(self, data_file):
        """
        获取分组信息和真实达标时间
        
        Args:
            data_file: 数据文件路径
            
        Returns:
            tuple: (bmi_groups DataFrame, true_times array)
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_strategy_name(self):
        """获取策略名称"""
        raise NotImplementedError("子类必须实现此方法")


class IntervalCensoredStrategy(GroupingStrategy):
    """
    区间删失生存模型分组策略（原有方法）
    """
    
    def get_groups_and_true_times(self, data_file):
        """使用区间删失生存模型进行分组"""
        from interval_censored_survival_model import IntervalCensoredSurvivalModel
        
        model = IntervalCensoredSurvivalModel()
        model.load_and_prepare_data(data_file)
        model.fit_aft_model(distribution='lognormal')  # 使用支持的分布
        model.perform_bmi_clustering(n_clusters=4)
        model.determine_optimal_timing(success_rate=0.9)
        
        bmi_groups = model.bmi_groups.copy()
        optimal_timings = model.optimal_timings.copy()
        true_times = optimal_timings.sort_values('组别')['最佳时点'].values
        
        return bmi_groups.sort_values('组别'), true_times
    
    def get_strategy_name(self):
        return "区间删失生存模型"


class BMIBoundaryStrategy(GroupingStrategy):
    """
    BMI分界值分组策略（新方法）
    """
    
    def get_groups_and_true_times(self, data_file):
        """使用BMI分界值进行分组"""
        # 导入新的NIPT时点优化器
        import sys
        import os
        
        # 添加目录3到路径，以便导入NIPTTimingOptimizer
        dir3_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '3')
        if dir3_path not in sys.path:
            sys.path.insert(0, dir3_path)
        
        from nipt_timing_optimization import NIPTTimingOptimizer
        
        # 使用NIPT优化器进行分析
        optimizer = NIPTTimingOptimizer(data_file)
        recommendations_df = optimizer.run_analysis()
        
        if recommendations_df is None or len(recommendations_df) == 0:
            raise ValueError("BMI分界值分组失败，无有效分组结果")
        
        # 转换为与原有格式兼容的分组信息
        bmi_groups = []
        true_times = []
        
        for _, row in recommendations_df.iterrows():
            bmi_groups.append({
                '组别': int(row['BMI组别']),
                'BMI区间': row['BMI范围_标准化'],
                'BMI均值': row['BMI均值_标准化'],
                '样本数': row['样本数']
            })
            true_times.append(row['建议检测时点_原始周数'])
        
        bmi_groups_df = pd.DataFrame(bmi_groups)
        true_times_array = np.array(true_times)
        
        return bmi_groups_df, true_times_array
    
    def get_strategy_name(self):
        return "BMI分界值分组"


class DetectionErrorAnalyzer:
    """
    NIPT检测误差分析器
    """
    
    def __init__(self, output_dir='.', threshold_logit=-3.178054):
        """
        初始化检测误差分析器
        
        Args:
            output_dir: 主输出目录
            threshold_logit: 4%浓度的logit变换值
        """
        self.output_dir = output_dir
        self.error_dir = os.path.join(output_dir, 'detection_error_analysis')
        self.threshold_logit = threshold_logit
        self.threshold_percentage = 0.04
        
        # 创建子目录
        os.makedirs(self.error_dir, exist_ok=True)
        
        # 分析结果存储
        self.noise_metrics = {}
        self.error_rates = {}
        self.simulation_results = {}
        # 报告上下文（用于动态描述分析依据/策略）
        self.report_context = {}
        
    def estimate_measurement_noise(self, df, col_woman='孕妇代码', col_y='Y染色体浓度', 
                                 col_week=None, smoothing_window=3):
        """
        估计测量噪声水平
        
        Args:
            df: 逐次测量数据
            col_woman: 孕妇代码列名
            col_y: Y染色体浓度列名
            col_week: 孕周列名
            smoothing_window: 平滑窗口大小
            
        Returns:
            dict: 噪声估计结果
        """
        print("=== 测量噪声估计 ===")
        
        noise_data = []
        
        for woman_code, woman_data in df.groupby(col_woman):
            if len(woman_data) < 3:  # 至少需要3个测量点
                continue
                
            # 按孕周排序（如果有）
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
            
            # 方法2: 相邻差分估计
            if len(y_values) >= 2:
                diffs = np.diff(y_values)
                # 使用MAD (Median Absolute Deviation) 估计噪声
                noise_diff = np.median(np.abs(diffs - np.median(diffs))) / 0.6745
            else:
                noise_diff = np.nan
                
            # 方法3: 线性拟合残差
            if len(y_values) >= 3:
                x_indices = np.arange(len(y_values))
                try:
                    # 线性拟合
                    coeffs = np.polyfit(x_indices, y_values, 1)
                    y_fit = np.polyval(coeffs, x_indices)
                    residuals_linear = y_values - y_fit
                    noise_linear = np.std(residuals_linear)
                except:
                    noise_linear = np.nan
            else:
                noise_linear = np.nan
            
            # 计算信噪比
            signal_mean = np.mean(y_values)
            signal_std = np.std(y_values)
            
            noise_data.append({
                '孕妇代码': woman_code,
                '测量次数': len(y_values),
                '信号均值': signal_mean,
                '信号标准差': signal_std,
                '噪声_平滑': noise_smooth,
                '噪声_差分': noise_diff,
                '噪声_线性': noise_linear,
                'SNR_平滑': signal_std / noise_smooth if not np.isnan(noise_smooth) and noise_smooth > 0 else np.nan,
                'SNR_差分': signal_std / noise_diff if not np.isnan(noise_diff) and noise_diff > 0 else np.nan,
                'SNR_线性': signal_std / noise_linear if not np.isnan(noise_linear) and noise_linear > 0 else np.nan
            })
        
        noise_df = pd.DataFrame(noise_data)
        
        # 汇总统计
        noise_summary = {
            '样本数': len(noise_df),
            '平均测量次数': noise_df['测量次数'].mean(),
            '噪声水平': {
                '平滑法': {
                    '均值': noise_df['噪声_平滑'].mean(),
                    '中位数': noise_df['噪声_平滑'].median(),
                    '标准差': noise_df['噪声_平滑'].std()
                },
                '差分法': {
                    '均值': noise_df['噪声_差分'].mean(),
                    '中位数': noise_df['噪声_差分'].median(),
                    '标准差': noise_df['噪声_差分'].std()
                },
                '线性法': {
                    '均值': noise_df['噪声_线性'].mean(),
                    '中位数': noise_df['噪声_线性'].median(),
                    '标准差': noise_df['噪声_线性'].std()
                }
            },
            '信噪比': {
                'SNR_平滑': noise_df['SNR_平滑'].mean(),
                'SNR_差分': noise_df['SNR_差分'].mean(),
                'SNR_线性': noise_df['SNR_线性'].mean()
            }
        }
        
        # 保存详细数据
        noise_df.to_csv(os.path.join(self.error_dir, 'measurement_noise_details.csv'), 
                       index=False, encoding='utf-8')
        
        self.noise_metrics = noise_summary
        
        print(f"噪声估计完成，共分析 {len(noise_df)} 个样本")
        print(f"平均噪声水平（平滑法）: {noise_summary['噪声水平']['平滑法']['均值']:.6f}")
        print(f"平均信噪比（平滑法）: {noise_summary['信噪比']['SNR_平滑']:.2f}")
        
        return noise_summary, noise_df
    
    def analyze_detection_errors(self, df, col_woman='孕妇代码', col_y='Y染色体浓度',
                               noise_level=None):
        """
        分析检测误差：假阳性和假阴性概率
        
        Args:
            df: 逐次测量数据
            col_woman: 孕妇代码列名
            col_y: Y染色体浓度列名
            noise_level: 噪声水平，如果为None则使用估计值
            
        Returns:
            dict: 误差分析结果
        """
        print("=== 检测误差分析 ===")
        
        if noise_level is None:
            # 使用之前估计的噪声水平
            if self.noise_metrics:
                noise_level = self.noise_metrics['噪声水平']['平滑法']['均值']
            else:
                noise_level = 0.1  # 默认噪声水平
        
        print(f"使用噪声水平: {noise_level:.6f}")
        
        # 分析不同浓度水平的误差率
        y_range = np.linspace(-5, 0, 100)  # logit变换后的浓度范围
        
        error_analysis = []
        
        for true_conc in y_range:
            # 假设真实浓度为true_conc，噪声为正态分布
            # 测量值 = 真实值 + 噪声
            
            # 计算假阳性率：真实浓度 < 阈值，但测量值 >= 阈值
            if true_conc < self.threshold_logit:
                # P(测量值 >= 阈值 | 真实值 < 阈值)
                false_positive_prob = 1 - stats.norm.cdf(
                    (self.threshold_logit - true_conc) / noise_level
                )
            else:
                false_positive_prob = 0.0
            
            # 计算假阴性率：真实浓度 >= 阈值，但测量值 < 阈值
            if true_conc >= self.threshold_logit:
                # P(测量值 < 阈值 | 真实值 >= 阈值)
                false_negative_prob = stats.norm.cdf(
                    (self.threshold_logit - true_conc) / noise_level
                )
            else:
                false_negative_prob = 0.0
            
            # 计算正确检测概率
            correct_prob = 1 - false_positive_prob - false_negative_prob
            
            error_analysis.append({
                '真实浓度_logit': true_conc,
                '真实浓度_百分比': 1 / (1 + np.exp(-true_conc)),  # 反logit变换
                '假阳性概率': false_positive_prob,
                '假阴性概率': false_negative_prob,
                '正确检测概率': correct_prob,
                '距离阈值': abs(true_conc - self.threshold_logit)
            })
        
        error_df = pd.DataFrame(error_analysis)
        
        # 计算关键统计量
        # 在阈值附近（±2个噪声标准差）的误差率
        near_threshold = error_df[
            error_df['距离阈值'] <= 2 * noise_level
        ]
        
        error_summary = {
            '噪声水平': noise_level,
            '阈值附近误差': {
                '范围': f"±{2*noise_level:.4f} (logit单位)",
                '平均假阳性率': near_threshold['假阳性概率'].mean(),
                '平均假阴性率': near_threshold['假阴性概率'].mean(),
                '最大假阳性率': near_threshold['假阳性概率'].max(),
                '最大假阴性率': near_threshold['假阴性概率'].max()
            },
            '4%阈值处误差': {
                '假阳性率': error_df.loc[
                    np.argmin(np.abs(error_df['真实浓度_logit'] - self.threshold_logit)), 
                    '假阳性概率'
                ],
                '假阴性率': error_df.loc[
                    np.argmin(np.abs(error_df['真实浓度_logit'] - self.threshold_logit)), 
                    '假阴性概率'
                ]
            }
        }
        
        # 保存误差分析数据
        error_df.to_csv(os.path.join(self.error_dir, 'detection_error_analysis.csv'), 
                       index=False, encoding='utf-8')
        
        self.error_rates = error_summary
        
        print(f"阈值附近平均假阳性率: {error_summary['阈值附近误差']['平均假阳性率']:.4f}")
        print(f"阈值附近平均假阴性率: {error_summary['阈值附近误差']['平均假阴性率']:.4f}")
        
        return error_summary, error_df
    
    def monte_carlo_simulation(self, true_times, bmi_groups, n_simulations=1000,
                             noise_level=None):
        """
        蒙特卡洛模拟检测误差对达标时间估计的影响
        
        Args:
            true_times: 各BMI组真实达标时间
            bmi_groups: BMI分组信息
            n_simulations: 模拟次数
            noise_level: 噪声水平
            
        Returns:
            dict: 模拟结果
        """
        print("=== 蒙特卡洛模拟 ===")
        
        if noise_level is None:
            if self.noise_metrics:
                noise_level = self.noise_metrics['噪声水平']['平滑法']['均值']
            else:
                noise_level = 0.1
        
        print(f"模拟参数: 噪声水平={noise_level:.6f}, 模拟次数={n_simulations}")
        
        simulation_results = []
        
        for i, (_, group) in enumerate(bmi_groups.iterrows()):
            true_time = true_times[i] if i < len(true_times) else 15.0
            group_name = f"第{group['组别']}组"
            
            print(f"模拟 {group_name} (真实达标时间: {true_time:.1f}周)")
            
            # 模拟多次检测的达标时间偏差
            detected_times = []
            
            for sim in range(n_simulations):
                # 模拟检测序列：假设每周检测一次，从10周开始
                start_week = 10
                max_week = 30
                weeks = np.arange(start_week, max_week + 1)
                
                # 生成真实浓度轨迹（假设线性增长）
                # 在true_time周达到阈值
                if true_time <= start_week:
                    # 如果真实达标时间很早，假设从开始就达标
                    true_concentrations = np.full(len(weeks), self.threshold_logit + 0.5)
                else:
                    # 线性增长到达标时间
                    slope = (self.threshold_logit - (self.threshold_logit - 1)) / (true_time - start_week)
                    true_concentrations = (self.threshold_logit - 1) + slope * (weeks - start_week)
                    true_concentrations = np.maximum(true_concentrations, self.threshold_logit - 1)
                
                # 添加测量噪声
                measured_concentrations = true_concentrations + np.random.normal(0, noise_level, len(weeks))
                
                # 找到首次达标的时间
                qualified_indices = np.where(measured_concentrations >= self.threshold_logit)[0]
                
                if len(qualified_indices) > 0:
                    detected_time = weeks[qualified_indices[0]]
                else:
                    detected_time = max_week  # 未达标，记为最大时间
                
                detected_times.append(detected_time)
            
            detected_times = np.array(detected_times)
            
            # 计算统计量
            bias = np.mean(detected_times) - true_time
            rmse = np.sqrt(np.mean((detected_times - true_time)**2))
            mae = np.mean(np.abs(detected_times - true_time))
            
            # 计算置信区间
            ci_lower = np.percentile(detected_times, 2.5)
            ci_upper = np.percentile(detected_times, 97.5)
            
            simulation_results.append({
                '组别': group['组别'],
                'BMI区间': group['BMI区间'],
                '真实达标时间': true_time,
                '预测均值': np.mean(detected_times),
                '预测中位数': np.median(detected_times),
                '偏差': bias,
                'RMSE': rmse,
                'MAE': mae,
                '标准差': np.std(detected_times),
                'CI_2.5%': ci_lower,
                'CI_97.5%': ci_upper,
                '检测成功率': np.mean(detected_times < max_week)
            })
        
        sim_df = pd.DataFrame(simulation_results)
        
        # 保存模拟结果
        sim_df.to_csv(os.path.join(self.error_dir, 'monte_carlo_simulation.csv'), 
                     index=False, encoding='utf-8')
        
        # 汇总统计
        summary = {
            '总体偏差': {
                '平均偏差': sim_df['偏差'].mean(),
                '平均RMSE': sim_df['RMSE'].mean(),
                '平均MAE': sim_df['MAE'].mean()
            },
            '各组结果': sim_df.to_dict('records')
        }
        
        self.simulation_results = summary
        
        print(f"模拟完成，平均时间偏差: {summary['总体偏差']['平均偏差']:.2f}周")
        print(f"平均RMSE: {summary['总体偏差']['平均RMSE']:.2f}周")
        
        return summary, sim_df


def create_error_visualizations(analyzer, noise_df, error_df, sim_df):
    """
    创建检测误差分析的可视化图表
    
    Args:
        analyzer: DetectionErrorAnalyzer实例
        noise_df: 噪声分析数据
        error_df: 误差分析数据
        sim_df: 模拟结果数据
    """
    print("=== 创建误差分析可视化 ===")
    
    # 设置中文字体
    set_chinese_font()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 噪声水平分布
    ax1 = axes[0, 0]
    noise_methods = ['噪声_平滑', '噪声_差分', '噪声_线性']
    noise_values = [noise_df[col].dropna() for col in noise_methods]
    
    ax1.boxplot(noise_values, labels=['平滑法', '差分法', '线性法'])
    ax1.set_ylabel('噪声水平')
    ax1.set_title('不同方法估计的噪声水平分布')
    ax1.grid(True, alpha=0.3)
    
    # 2. 信噪比分布
    ax2 = axes[0, 1]
    snr_methods = ['SNR_平滑', 'SNR_差分', 'SNR_线性']
    snr_values = [noise_df[col].dropna() for col in snr_methods]
    
    ax2.boxplot(snr_values, labels=['平滑法', '差分法', '线性法'])
    ax2.set_ylabel('信噪比')
    ax2.set_title('不同方法计算的信噪比分布')
    ax2.grid(True, alpha=0.3)
    
    # 3. 假阳性/假阴性概率曲线
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
    
    # 4. 达标时间偏差
    ax4 = axes[1, 0]
    groups = [f"第{row['组别']}组" for _, row in sim_df.iterrows()]
    biases = sim_df['偏差'].values
    
    colors = ['red' if b > 0 else 'blue' for b in biases]
    bars = ax4.bar(groups, biases, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_ylabel('时间偏差 (周)')
    ax4.set_title('各BMI组达标时间估计偏差')
    ax4.grid(True, alpha=0.3)
    
    # 在柱子上标注数值
    for bar, bias in zip(bars, biases):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.1 if height > 0 else height - 0.1,
                f'{bias:.1f}', ha='center', va='bottom' if height > 0 else 'top')
    
    # 5. RMSE和MAE对比
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
    
    # 6. 置信区间图
    ax6 = axes[1, 2]
    true_times = sim_df['真实达标时间']
    pred_means = sim_df['预测均值']
    ci_lower = sim_df['CI_2.5%']
    ci_upper = sim_df['CI_97.5%']
    
    # 绘制置信区间（确保误差值为正）
    for i, group in enumerate(groups):
        lower_err = max(0, pred_means.iloc[i] - ci_lower.iloc[i])
        upper_err = max(0, ci_upper.iloc[i] - pred_means.iloc[i])
        
        ax6.errorbar(i, pred_means.iloc[i], 
                    yerr=[[lower_err], [upper_err]], 
                    fmt='o', capsize=5, label='预测' if i == 0 else "")
        ax6.scatter(i, true_times.iloc[i], color='red', marker='x', s=100, 
                   label='真实' if i == 0 else "")
    
    ax6.set_xlabel('BMI组')
    ax6.set_ylabel('达标时间 (周)')
    ax6.set_title('达标时间预测置信区间')
    ax6.set_xticks(range(len(groups)))
    ax6.set_xticklabels(groups)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analyzer.error_dir, 'detection_error_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"误差分析图表已保存为: {os.path.join(analyzer.error_dir, 'detection_error_analysis.png')}")


def generate_error_report(analyzer):
    """
    生成检测误差分析报告
    
    Args:
        analyzer: DetectionErrorAnalyzer实例
    """
    print("=== 生成误差分析报告 ===")
    
    # 动态设置报告标题与分析依据描述
    analysis_basis = analyzer.report_context.get('analysis_basis', '区间删失生存模型分组')
    report_title = analyzer.report_context.get('report_title', 'NIPT检测误差分析报告')
    report = f"""# {report_title}

## 1. 分析概述

本报告分析NIPT检测过程中的测量误差及其对达标时间估计的影响，基于{analysis_basis}的结果进行误差传播分析。
"""

    # 可选：分组设定与依据
    group_summary = analyzer.report_context.get('group_summary')
    if group_summary:
        report += f"""

## 1.1 分组设定与依据

{group_summary}
"""

    report += f"""

## 2. 测量噪声分析

### 2.1 噪声水平估计
"""
    
    if analyzer.noise_metrics:
        noise = analyzer.noise_metrics
        report += f"""
- **样本数量**: {noise['样本数']}个孕妇
- **平均测量次数**: {noise['平均测量次数']:.1f}次

#### 不同估计方法的噪声水平：
- **平滑法**: 均值={noise['噪声水平']['平滑法']['均值']:.6f}, 中位数={noise['噪声水平']['平滑法']['中位数']:.6f}
- **差分法**: 均值={noise['噪声水平']['差分法']['均值']:.6f}, 中位数={noise['噪声水平']['差分法']['中位数']:.6f}  
- **线性法**: 均值={noise['噪声水平']['线性法']['均值']:.6f}, 中位数={noise['噪声水平']['线性法']['中位数']:.6f}

#### 信噪比分析：
- **SNR (平滑法)**: {noise['信噪比']['SNR_平滑']:.2f}
- **SNR (差分法)**: {noise['信噪比']['SNR_差分']:.2f}
- **SNR (线性法)**: {noise['信噪比']['SNR_线性']:.2f}
"""

    report += f"""
## 3. 检测误差分析

### 3.1 假阳性/假阴性概率
"""
    
    if analyzer.error_rates:
        error = analyzer.error_rates
        report += f"""
- **使用噪声水平**: {error['噪声水平']:.6f}

#### 4%阈值处的误差：
- **假阳性率**: {error['4%阈值处误差']['假阳性率']:.4f} ({error['4%阈值处误差']['假阳性率']*100:.2f}%)
- **假阴性率**: {error['4%阈值处误差']['假阴性率']:.4f} ({error['4%阈值处误差']['假阴性率']*100:.2f}%)

#### 阈值附近区域的误差：
- **范围**: {error['阈值附近误差']['范围']}
- **平均假阳性率**: {error['阈值附近误差']['平均假阳性率']:.4f} ({error['阈值附近误差']['平均假阳性率']*100:.2f}%)
- **平均假阴性率**: {error['阈值附近误差']['平均假阴性率']:.4f} ({error['阈值附近误差']['平均假阴性率']*100:.2f}%)
- **最大假阳性率**: {error['阈值附近误差']['最大假阳性率']:.4f} ({error['阈值附近误差']['最大假阳性率']*100:.2f}%)
- **最大假阴性率**: {error['阈值附近误差']['最大假阴性率']:.4f} ({error['阈值附近误差']['最大假阴性率']*100:.2f}%)
"""

    report += f"""
## 4. 蒙特卡洛模拟结果

### 4.1 达标时间估计偏差
"""
    
    if analyzer.simulation_results:
        sim = analyzer.simulation_results
        report += f"""
#### 总体误差指标：
- **平均时间偏差**: {sim['总体偏差']['平均偏差']:.2f}周
- **平均RMSE**: {sim['总体偏差']['平均RMSE']:.2f}周  
- **平均MAE**: {sim['总体偏差']['平均MAE']:.2f}周

#### 各BMI组详细结果：
"""
        
        for group_result in sim['各组结果']:
            report += f"""
##### {group_result['BMI区间']}
- **真实达标时间**: {group_result['真实达标时间']:.1f}周
- **预测均值**: {group_result['预测均值']:.1f}周
- **时间偏差**: {group_result['偏差']:+.1f}周
- **RMSE**: {group_result['RMSE']:.1f}周
- **95%置信区间**: [{group_result['CI_2.5%']:.1f}, {group_result['CI_97.5%']:.1f}]周
- **检测成功率**: {group_result['检测成功率']*100:.1f}%
"""

    report += f"""
## 5. 结论与建议

### 5.1 主要发现
1. **噪声水平**: 测量噪声主要来源于设备精度和生物变异，不同估计方法结果基本一致
2. **误判风险**: 在4%阈值附近存在一定的假阳性和假阴性风险，需要考虑在临床决策中
3. **时间偏差**: 检测误差会导致达标时间估计出现系统性偏差，特别是对于接近阈值的情况

### 5.2 临床建议
1. **重复检测**: 对于接近阈值的检测结果，建议进行重复检测以降低误判风险
2. **动态监测**: 采用多时点检测策略，利用时间序列信息提高检测准确性
3. **个体化阈值**: 考虑基于个体BMI和临床特征调整检测阈值

### 5.3 方法改进
1. **噪声建模**: 可考虑更复杂的噪声模型，如时间相关噪声
2. **贝叶斯方法**: 采用贝叶斯统计方法整合多次检测信息
3. **机器学习**: 利用深度学习方法学习浓度变化模式，提高预测精度

## 6. 技术说明

- **分析基础**: 基于{analyzer.report_context.get('analysis_basis', '区间删失生存模型分组')}的检测数据
- **噪声估计**: 采用多种方法交叉验证噪声水平
- **误差分析**: 基于正态噪声假设的理论分析
- **模拟验证**: 蒙特卡洛方法验证误差传播效应

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存报告
    report_path = os.path.join(analyzer.error_dir, 'detection_error_analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"误差分析报告已保存为: {report_path}")
    
    return report


def run_error_analysis_with_strategy(strategy, data_file=None, output_suffix=''):
    """
    使用指定策略运行检测误差分析
    
    Args:
        strategy: GroupingStrategy子类实例
        data_file: 数据文件路径，如果为None则使用默认路径
        output_suffix: 输出目录后缀，用于区分不同策略的结果
    """
    print(f"=== NIPT检测误差分析 - {strategy.get_strategy_name()} ===")
    
    # 设置输出目录
    if output_suffix:
        output_dir = os.path.join(SCRIPT_DIR, f'detection_error_results_{output_suffix}')
    else:
        output_dir = os.path.join(SCRIPT_DIR, 'detection_error_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 默认数据文件路径
    if data_file is None:
        data_file = os.path.join(SCRIPT_DIR, 'processed_data.csv')
    
    # 1) 读取数据
    raw_df = pd.read_csv(data_file)
    
    # 2) 估计噪声
    analyzer = DetectionErrorAnalyzer(output_dir=output_dir)
    # 设置报告上下文（用于模板动态描述）
    analyzer.report_context['analysis_basis'] = strategy.get_strategy_name()
    noise_summary, noise_df = analyzer.estimate_measurement_noise(
        raw_df,
        col_woman='孕妇代码',
        col_y='Y染色体浓度',
        col_week='孕周_标准化',
        smoothing_window=3,
    )
    
    # 3) 误差率分析
    error_summary, error_df = analyzer.analyze_detection_errors(
        raw_df,
        col_woman='孕妇代码',
        col_y='Y染色体浓度',
        noise_level=noise_summary['噪声水平']['平滑法']['均值'] if '噪声水平' in noise_summary else None,
    )
    
    # 4) 使用指定策略获取分组和真实时间
    try:
        bmi_groups, true_times = strategy.get_groups_and_true_times(data_file)
        print(f"使用{strategy.get_strategy_name()}获取分组成功")
        print(f"分组数量: {len(bmi_groups)}")
        print(f"真实时间: {true_times}")
    except Exception as e:
        print(f"使用{strategy.get_strategy_name()}获取分组失败: {e}")
        return None
    
    # 5) 蒙特卡洛模拟
    sim_summary, sim_df = analyzer.monte_carlo_simulation(
        true_times=true_times,
        bmi_groups=bmi_groups,
        n_simulations=1000,
        noise_level=noise_summary['噪声水平']['平滑法']['均值'] if '噪声水平' in noise_summary else None,
    )
    
    # 6) 可视化与报告
    create_error_visualizations(analyzer, noise_df, error_df, sim_df)
    generate_error_report(analyzer)
    
    print(f"\n=== {strategy.get_strategy_name()}检测误差分析完成 ===")
    print("所有结果已保存到子目录:", os.path.relpath(analyzer.error_dir, SCRIPT_DIR))
    
    return {
        'strategy': strategy.get_strategy_name(),
        'analyzer': analyzer,
        'noise_summary': noise_summary,
        'error_summary': error_summary,
        'sim_summary': sim_summary
    }


def main(strategy_type='interval_censored'):
    """
    主函数：运行检测误差分析
    
    Args:
        strategy_type: 分组策略类型
            - 'interval_censored': 区间删失生存模型（原有方法）
            - 'bmi_boundary': BMI分界值分组（新方法）
    """
    data_file = os.path.join(SCRIPT_DIR, 'processed_data.csv')
    
    if strategy_type == 'interval_censored':
        strategy = IntervalCensoredStrategy()
        suffix = 'interval_censored'
    elif strategy_type == 'bmi_boundary':
        strategy = BMIBoundaryStrategy()
        suffix = 'bmi_boundary'
    else:
        raise ValueError(f"不支持的策略类型: {strategy_type}")
    
    return run_error_analysis_with_strategy(strategy, data_file, suffix)


def compare_strategies():
    """
    比较不同分组策略的检测误差分析结果
    """
    print("=== 比较不同分组策略 ===")
    
    strategies = [
        ('interval_censored', IntervalCensoredStrategy()),
        ('bmi_boundary', BMIBoundaryStrategy())
    ]
    
    results = {}
    data_file = os.path.join(SCRIPT_DIR, 'processed_data.csv')
    
    for strategy_name, strategy in strategies:
        try:
            print(f"\n--- 运行{strategy.get_strategy_name()}策略 ---")
            result = run_error_analysis_with_strategy(strategy, data_file, strategy_name)
            if result:
                results[strategy_name] = result
        except Exception as e:
            print(f"策略{strategy.get_strategy_name()}执行失败: {e}")
            continue
    
    # 生成比较报告
    if len(results) > 1:
        print("\n=== 策略比较总结 ===")
        for strategy_name, result in results.items():
            noise_level = result['noise_summary']['噪声水平']['平滑法']['均值'] if 'noise_summary' in result else 'N/A'
            avg_bias = result['sim_summary']['总体偏差']['平均偏差'] if 'sim_summary' in result else 'N/A'
            print(f"{result['strategy']}:")
            print(f"  平均噪声水平: {noise_level:.6f}" if isinstance(noise_level, float) else f"  平均噪声水平: {noise_level}")
            print(f"  平均时间偏差: {avg_bias:.2f}周" if isinstance(avg_bias, float) else f"  平均时间偏差: {avg_bias}")
            print()
    
    return results


if __name__ == '__main__':
    main()
