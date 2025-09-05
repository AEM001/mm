#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测误差分析：分析Y染色体浓度和BMI测量误差对聚类和最佳时点的影响
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys
import warnings
warnings.filterwarnings('ignore')

# 确保可以导入项目根目录的模块
sys.path.append('/Users/Mac/Downloads/mm')
from set_chinese_font import set_chinese_font
set_chinese_font()

def add_random_error(data, error_rate=0.05):
    """
    为数据添加随机误差
    
    Parameters:
    data: 原始数据
    error_rate: 误差率，默认5%
    
    Returns:
    添加误差后的数据
    """
    # 生成[-1, 1]区间的均匀随机数
    random_factors = np.random.uniform(-1, 1, len(data))
    # 应用误差公式：新值 = 原值 × (1 + 误差率 × 随机因子)
    perturbed_data = data * (1 + error_rate * random_factors)
    return perturbed_data

def calculate_optimal_timing(cluster_data):
    """计算聚类组的最佳NIPT时点"""
    gestational_weeks = cluster_data['最早达标孕周'].values
    t_10 = np.percentile(gestational_weeks, 10)
    t_20 = np.percentile(gestational_weeks, 20)
    
    # 应用12周风险阈值逻辑
    if t_10 > 12.0:
        if t_20 <= 12.5:
            optimal_timing = 12.0
            coverage_rate = 80
        else:
            optimal_timing = t_10
            coverage_rate = 90
    else:
        optimal_timing = t_10
        coverage_rate = 90
    
    return optimal_timing, coverage_rate, t_10, t_20

def perform_single_simulation(original_data, y_error=True, bmi_error=False, error_rate=0.05):
    """执行单次误差模拟"""
    df = original_data.copy()
    
    # 添加Y染色体浓度误差
    if y_error:
        df['Y染色体浓度'] = add_random_error(df['Y染色体浓度'], error_rate)
    
    # 添加BMI误差
    if bmi_error:
        df['BMI'] = add_random_error(df['BMI'], error_rate)
    
    # 重新筛选Y染色体浓度>=4%的数据
    if y_error:
        df = df[df['Y染色体浓度'] >= 0.04].copy()
    
    # 如果数据量太少，返回None
    if len(df) < 10:
        return None
    
    # 进行聚类分析
    features = df[['BMI', '最早达标孕周']].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 使用k=2进行聚类（基于原始分析结果）
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    df['聚类标签'] = cluster_labels
    
    # 计算每组的最佳时点
    results = []
    for i in range(2):
        cluster_data = df[df['聚类标签'] == i]
        if len(cluster_data) > 0:
            optimal_timing, coverage_rate, t_10, t_20 = calculate_optimal_timing(cluster_data)
            results.append({
                '聚类组': i + 1,
                '样本数': len(cluster_data),
                'BMI均值': cluster_data['BMI'].mean(),
                '最佳NIPT时点': optimal_timing,
                '10%分位数': t_10,
                '20%分位数': t_20
            })
    
    return results

def error_analysis():
    """执行误差分析"""
    
    print("开始检测误差分析...")
    
    # 1. 读取原始数据和聚类结果
    print("1. 读取原始数据...")
    original_data = pd.read_csv('/Users/Mac/Downloads/mm/2/processed_data_problem2.csv')
    original_timing = pd.read_csv('/Users/Mac/Downloads/mm/2/results/optimal_timing_summary.csv')
    
    print(f"原始数据量: {len(original_data)} 名孕妇")
    print("原始最佳时点:")
    print(original_timing[['聚类组', 'BMI范围', '最佳NIPT时点']].to_string(index=False))
    
    # 2. Y染色体浓度误差分析
    print("\n2. Y染色体浓度误差分析 (±5%误差)...")
    
    n_simulations = 1000
    y_error_results = []
    
    print(f"执行 {n_simulations} 次Monte Carlo模拟...")
    
    for i in range(n_simulations):
        if (i + 1) % 200 == 0:
            print(f"  完成 {i + 1}/{n_simulations} 次模拟")
        
        result = perform_single_simulation(original_data, y_error=True, bmi_error=False)
        if result is not None:
            y_error_results.append(result)
    
    # 分析Y染色体浓度误差结果
    print(f"\n有效模拟次数: {len(y_error_results)}")
    
    # 提取各组的最佳时点分布
    group1_timings = []
    group2_timings = []
    
    for result in y_error_results:
        for group in result:
            if group['聚类组'] == 1:
                group1_timings.append(group['最佳NIPT时点'])
            elif group['聚类组'] == 2:
                group2_timings.append(group['最佳NIPT时点'])
    
    # 3. BMI误差分析
    print("\n3. BMI误差分析 (±5%误差)...")
    
    bmi_error_results = []
    
    for i in range(n_simulations):
        if (i + 1) % 200 == 0:
            print(f"  完成 {i + 1}/{n_simulations} 次模拟")
        
        result = perform_single_simulation(original_data, y_error=False, bmi_error=True)
        if result is not None:
            bmi_error_results.append(result)
    
    print(f"有效模拟次数: {len(bmi_error_results)}")
    
    # 提取BMI误差的各组最佳时点分布
    bmi_group1_timings = []
    bmi_group2_timings = []
    
    for result in bmi_error_results:
        for group in result:
            if group['聚类组'] == 1:
                bmi_group1_timings.append(group['最佳NIPT时点'])
            elif group['聚类组'] == 2:
                bmi_group2_timings.append(group['最佳NIPT时点'])
    
    # 4. 统计分析和可视化
    print("\n4. 统计分析结果")
    print("="*60)
    
    # Y染色体浓度误差统计
    if group1_timings and group2_timings:
        print("Y染色体浓度误差影响:")
        print(f"第1组最佳时点: 原始={original_timing.iloc[0]['最佳NIPT时点']:.1f}周")
        print(f"  误差后均值: {np.mean(group1_timings):.1f} ± {np.std(group1_timings):.1f}周")
        print(f"  变化范围: {np.min(group1_timings):.1f} - {np.max(group1_timings):.1f}周")
        
        print(f"第2组最佳时点: 原始={original_timing.iloc[1]['最佳NIPT时点']:.1f}周")
        print(f"  误差后均值: {np.mean(group2_timings):.1f} ± {np.std(group2_timings):.1f}周")
        print(f"  变化范围: {np.min(group2_timings):.1f} - {np.max(group2_timings):.1f}周")
    
    # BMI误差统计
    if bmi_group1_timings and bmi_group2_timings:
        print("\nBMI误差影响:")
        print(f"第1组最佳时点: 原始={original_timing.iloc[0]['最佳NIPT时点']:.1f}周")
        print(f"  误差后均值: {np.mean(bmi_group1_timings):.1f} ± {np.std(bmi_group1_timings):.1f}周")
        print(f"  变化范围: {np.min(bmi_group1_timings):.1f} - {np.max(bmi_group1_timings):.1f}周")
        
        print(f"第2组最佳时点: 原始={original_timing.iloc[1]['最佳NIPT时点']:.1f}周")
        print(f"  误差后均值: {np.mean(bmi_group2_timings):.1f} ± {np.std(bmi_group2_timings):.1f}周")
        print(f"  变化范围: {np.min(bmi_group2_timings):.1f} - {np.max(bmi_group2_timings):.1f}周")
    
    # 5. 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Y染色体浓度误差 - 第1组
    if group1_timings:
        axes[0, 0].hist(group1_timings, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(original_timing.iloc[0]['最佳NIPT时点'], color='red', linestyle='--', linewidth=2, label='原始值')
        axes[0, 0].set_title('Y染色体浓度误差 - 第1组最佳时点分布')
        axes[0, 0].set_xlabel('最佳NIPT时点 (周)')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Y染色体浓度误差 - 第2组
    if group2_timings:
        axes[0, 1].hist(group2_timings, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].axvline(original_timing.iloc[1]['最佳NIPT时点'], color='red', linestyle='--', linewidth=2, label='原始值')
        axes[0, 1].set_title('Y染色体浓度误差 - 第2组最佳时点分布')
        axes[0, 1].set_xlabel('最佳NIPT时点 (周)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # BMI误差 - 第1组
    if bmi_group1_timings:
        axes[1, 0].hist(bmi_group1_timings, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].axvline(original_timing.iloc[0]['最佳NIPT时点'], color='red', linestyle='--', linewidth=2, label='原始值')
        axes[1, 0].set_title('BMI误差 - 第1组最佳时点分布')
        axes[1, 0].set_xlabel('最佳NIPT时点 (周)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # BMI误差 - 第2组
    if bmi_group2_timings:
        axes[1, 1].hist(bmi_group2_timings, bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].axvline(original_timing.iloc[1]['最佳NIPT时点'], color='red', linestyle='--', linewidth=2, label='原始值')
        axes[1, 1].set_title('BMI误差 - 第2组最佳时点分布')
        axes[1, 1].set_xlabel('最佳NIPT时点 (周)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/Mac/Downloads/mm/2/error_analysis/error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 保存详细结果
    error_summary = {
        'Y染色体浓度误差_第1组': {
            '原始值': original_timing.iloc[0]['最佳NIPT时点'],
            '误差后均值': np.mean(group1_timings) if group1_timings else None,
            '标准差': np.std(group1_timings) if group1_timings else None,
            '最小值': np.min(group1_timings) if group1_timings else None,
            '最大值': np.max(group1_timings) if group1_timings else None
        },
        'Y染色体浓度误差_第2组': {
            '原始值': original_timing.iloc[1]['最佳NIPT时点'],
            '误差后均值': np.mean(group2_timings) if group2_timings else None,
            '标准差': np.std(group2_timings) if group2_timings else None,
            '最小值': np.min(group2_timings) if group2_timings else None,
            '最大值': np.max(group2_timings) if group2_timings else None
        },
        'BMI误差_第1组': {
            '原始值': original_timing.iloc[0]['最佳NIPT时点'],
            '误差后均值': np.mean(bmi_group1_timings) if bmi_group1_timings else None,
            '标准差': np.std(bmi_group1_timings) if bmi_group1_timings else None,
            '最小值': np.min(bmi_group1_timings) if bmi_group1_timings else None,
            '最大值': np.max(bmi_group1_timings) if bmi_group1_timings else None
        },
        'BMI误差_第2组': {
            '原始值': original_timing.iloc[1]['最佳NIPT时点'],
            '误差后均值': np.mean(bmi_group2_timings) if bmi_group2_timings else None,
            '标准差': np.std(bmi_group2_timings) if bmi_group2_timings else None,
            '最小值': np.min(bmi_group2_timings) if bmi_group2_timings else None,
            '最大值': np.max(bmi_group2_timings) if bmi_group2_timings else None
        }
    }
    
    # 转换为DataFrame并保存
    error_df = pd.DataFrame(error_summary).T
    error_df.to_csv('/Users/Mac/Downloads/mm/2/error_analysis/error_analysis_summary.csv', encoding='utf-8-sig')
    
    # 保存原始数据分布用于对比
    if group1_timings and group2_timings:
        y_error_data = pd.DataFrame({
            '第1组_Y误差': pd.Series(group1_timings),
            '第2组_Y误差': pd.Series(group2_timings)
        })
        y_error_data.to_csv('/Users/Mac/Downloads/mm/2/error_analysis/y_concentration_error_distribution.csv', 
                           index=False, encoding='utf-8-sig')
    
    if bmi_group1_timings and bmi_group2_timings:
        bmi_error_data = pd.DataFrame({
            '第1组_BMI误差': pd.Series(bmi_group1_timings),
            '第2组_BMI误差': pd.Series(bmi_group2_timings)
        })
        bmi_error_data.to_csv('/Users/Mac/Downloads/mm/2/error_analysis/bmi_error_distribution.csv', 
                             index=False, encoding='utf-8-sig')
    
    print(f"\n误差分析结果已保存至: /Users/Mac/Downloads/mm/2/error_analysis/")
    print(f"  - error_analysis_summary.csv: 统计汇总")
    print(f"  - error_analysis.png: 分布图表")
    print(f"  - y_concentration_error_distribution.csv: Y浓度误差分布数据")
    print(f"  - bmi_error_distribution.csv: BMI误差分布数据")
    
    return error_summary

if __name__ == "__main__":
    error_results = error_analysis()
