#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-means聚类分析：对BMI和最早达标孕周进行聚类分组
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import sys

# 确保可以导入项目根目录的模块
sys.path.append('/Users/Mac/Downloads/mm')
from set_chinese_font import set_chinese_font
# 设置matplotlib以正确显示中文
set_chinese_font()

def perform_clustering():
    """执行K-means聚类分析"""
    
    # 1. 读取数据
    print("读取处理后的数据...")
    df = pd.read_csv('/Users/Mac/Downloads/mm/2/processed_data_problem2.csv')
    print(f"数据量: {len(df)} 名孕妇")
    
    # 2. 准备聚类数据
    print("\n准备聚类数据...")
    features = df[['BMI', '最早达标孕周']].copy()
    print("聚类特征统计:")
    print(features.describe())
    
    # 3. Z-score标准化
    print("\n进行Z-score标准化...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=['BMI_标准化', '最早达标孕周_标准化'])
    
    print("标准化后特征统计:")
    print(features_scaled_df.describe())
    
    # 4. 肘部法则确定最优k值
    print("\n使用肘部法则确定最优聚类数...")
    k_range = range(2, 11)
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))
    
    # 绘制肘部法则图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 组内误差平方和
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('聚类数 k')
    ax1.set_ylabel('组内误差平方和 (WCSS)')
    ax1.set_title('肘部法则 - WCSS')
    ax1.grid(True)
    
    # 轮廓系数
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('聚类数 k')
    ax2.set_ylabel('轮廓系数')
    ax2.set_title('轮廓系数分析')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/Mac/Downloads/mm/2/results/elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出各k值的评估指标
    print("\n各聚类数的评估指标:")
    for i, k in enumerate(k_range):
        print(f"k={k}: WCSS={inertias[i]:.2f}, 轮廓系数={silhouette_scores[i]:.3f}")
    
    # 5. 选择最优k值（基于轮廓系数）
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\n推荐聚类数: k={optimal_k} (轮廓系数最高: {max(silhouette_scores):.3f})")
    
    # 6. 执行最终聚类
    print(f"\n执行K-means聚类 (k={optimal_k})...")
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(features_scaled)
    
    # 7. 添加聚类结果到原数据
    df['聚类标签'] = cluster_labels
    df['BMI_标准化'] = features_scaled[:, 0]
    df['最早达标孕周_标准化'] = features_scaled[:, 1]
    
    # 8. 分析聚类结果
    print("\n聚类结果分析:")
    optimal_timing_results = []
    
    for i in range(optimal_k):
        cluster_data = df[df['聚类标签'] == i]
        print(f"\n第{i+1}组 (n={len(cluster_data)}):")
        print(f"  BMI范围: {cluster_data['BMI'].min():.1f} - {cluster_data['BMI'].max():.1f}")
        print(f"  BMI均值: {cluster_data['BMI'].mean():.1f} ± {cluster_data['BMI'].std():.1f}")
        print(f"  最早达标孕周范围: {cluster_data['最早达标孕周'].min():.1f} - {cluster_data['最早达标孕周'].max():.1f}")
        print(f"  最早达标孕周均值: {cluster_data['最早达标孕周'].mean():.1f} ± {cluster_data['最早达标孕周'].std():.1f}")
        
        # 计算最佳NIPT时点
        gestational_weeks = cluster_data['最早达标孕周'].values
        t_10 = np.percentile(gestational_weeks, 10)
        t_20 = np.percentile(gestational_weeks, 20)
        
        # 应用12周风险阈值逻辑
        if t_10 > 12.0:
            if t_20 <= 12.5:  # 略大于12周的情况
                optimal_timing = 12.0
                coverage_rate = 80  # 80%孕妇满足条件
                print(f"  10%分位数: {t_10:.1f}周 (>12周)")
                print(f"  20%分位数: {t_20:.1f}周")
                print(f"  最佳NIPT时点: {optimal_timing:.1f}周 (风险优先，覆盖率80%)")
            else:
                optimal_timing = t_10
                coverage_rate = 90  # 90%孕妇满足条件
                print(f"  10%分位数: {t_10:.1f}周")
                print(f"  最佳NIPT时点: {optimal_timing:.1f}周 (覆盖率90%)")
        else:
            optimal_timing = t_10
            coverage_rate = 90  # 90%孕妇满足条件
            print(f"  10%分位数: {t_10:.1f}周 (≤12周)")
            print(f"  最佳NIPT时点: {optimal_timing:.1f}周 (低风险区间，覆盖率90%)")
        
        optimal_timing_results.append({
            '聚类组': i + 1,
            '样本数': len(cluster_data),
            'BMI范围': f"{cluster_data['BMI'].min():.1f}-{cluster_data['BMI'].max():.1f}",
            'BMI均值': cluster_data['BMI'].mean(),
            '10%分位数': t_10,
            '20%分位数': t_20,
            '最佳NIPT时点': optimal_timing,
            '覆盖率': coverage_rate
        })
    
    # 9. 可视化聚类结果
    plt.figure(figsize=(12, 5))
    
    # 原始数据聚类图
    plt.subplot(1, 2, 1)
    colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))
    for i in range(optimal_k):
        cluster_data = df[df['聚类标签'] == i]
        plt.scatter(cluster_data['BMI'], cluster_data['最早达标孕周'], 
                   c=[colors[i]], label=f'第{i+1}组', alpha=0.7)
    
    plt.xlabel('BMI')
    plt.ylabel('最早达标孕周')
    plt.title('聚类结果 (原始数据)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 标准化数据聚类图
    plt.subplot(1, 2, 2)
    for i in range(optimal_k):
        cluster_data = df[df['聚类标签'] == i]
        plt.scatter(cluster_data['BMI_标准化'], cluster_data['最早达标孕周_标准化'], 
                   c=[colors[i]], label=f'第{i+1}组', alpha=0.7)
    
    # 绘制聚类中心
    centers = final_kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=200, linewidths=3, label='聚类中心')
    
    plt.xlabel('BMI (标准化)')
    plt.ylabel('最早达标孕周 (标准化)')
    plt.title('聚类结果 (标准化数据)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/Mac/Downloads/mm/2/results/clustering_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 10. 输出最佳NIPT时点汇总
    print("\n" + "="*60)
    print("最佳NIPT时点汇总")
    print("="*60)
    
    timing_summary_df = pd.DataFrame(optimal_timing_results)
    print(timing_summary_df.to_string(index=False, float_format='%.1f'))
    
    # 11. 保存结果
    output_file = '/Users/Mac/Downloads/mm/2/results/clustered_data.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n聚类结果已保存至: {output_file}")
    
    # 保存最佳时点汇总
    timing_file = '/Users/Mac/Downloads/mm/2/results/optimal_timing_summary.csv'
    timing_summary_df.to_csv(timing_file, index=False, encoding='utf-8-sig')
    print(f"最佳NIPT时点汇总已保存至: {timing_file}")
    
    return df, optimal_k, final_kmeans, scaler, timing_summary_df

if __name__ == "__main__":
    clustered_data, k, model, scaler, timing_summary = perform_clustering()
