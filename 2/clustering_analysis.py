#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-means聚类分析：BMI一维分组以生成不重叠的BMI区间，并计算各组最佳NIPT时点
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
    """执行K-means聚类分析（BMI一维分组，保证BMI区间不重叠）"""

    # 1. 读取数据
    print("读取处理后的数据...")
    df = pd.read_csv('/Users/Mac/Downloads/mm/2/processed_data_problem2.csv')
    print(f"数据量: {len(df)} 名孕妇")

    # 2. 准备数据视图（用于可视化的标准化，不用于聚类）
    print("\n准备特征数据...")
    features = df[['BMI', '最早达标孕周']].copy()
    print("特征统计:")
    print(features.describe())

    print("\n进行Z-score标准化（仅用于可视化）...")
    viz_scaler = StandardScaler()
    features_viz_scaled = viz_scaler.fit_transform(features)
    features_viz_scaled_df = pd.DataFrame(features_viz_scaled, columns=['BMI_标准化', '最早达标孕周_标准化'])
    df['BMI_标准化'] = features_viz_scaled_df['BMI_标准化']
    df['最早达标孕周_标准化'] = features_viz_scaled_df['最早达标孕周_标准化']

    print("标准化后特征统计（用于可视化）:")
    print(features_viz_scaled_df.describe())

    # 3. 基于BMI的一维K-means聚类，确保BMI区间不重叠
    print("\n使用BMI一维K-means进行分组（保证BMI区间不重叠）...")
    bmi_scaler = StandardScaler()
    bmi_scaled = bmi_scaler.fit_transform(df[['BMI']])  # shape (n_samples, 1)

    # 4. 肘部法则与轮廓系数确定最优k
    print("\n使用肘部法则与轮廓系数确定最优聚类数...")
    k_range = range(2, 11)
    inertias, silhouette_scores_list = [], []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(bmi_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores_list.append(silhouette_score(bmi_scaled, kmeans.labels_))

    # 绘制肘部法则图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 组内误差平方和
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('聚类数 k')
    ax1.set_ylabel('组内误差平方和 (WCSS)')
    ax1.set_title('肘部法则 - WCSS (BMI一维)')
    ax1.grid(True)

    # 轮廓系数
    ax2.plot(k_range, silhouette_scores_list, 'ro-')
    ax2.set_xlabel('聚类数 k')
    ax2.set_ylabel('轮廓系数')
    ax2.set_title('轮廓系数分析 (BMI一维)')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('/Users/Mac/Downloads/mm/2/results/elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n各聚类数的评估指标:")
    for i, k in enumerate(k_range):
        print(f"k={k}: WCSS={inertias[i]:.2f}, 轮廓系数={silhouette_scores_list[i]:.3f}")

    # 5. 选择最优k值（基于轮廓系数）
    optimal_k = k_range[np.argmax(silhouette_scores_list)]
    print(f"\n推荐聚类数: k={optimal_k} (轮廓系数最高: {max(silhouette_scores_list):.3f})")

    # 6. 执行最终聚类（BMI一维）
    print(f"\n执行BMI一维K-means聚类 (k={optimal_k})...")
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    raw_labels = final_kmeans.fit_predict(bmi_scaled)

    # 将聚类中心按BMI从低到高排序，并重映射标签，保证分组顺序与BMI区间一致
    centers_std = final_kmeans.cluster_centers_.reshape(-1)  # 标准化空间中的中心
    order = np.argsort(centers_std)
    label_map = {old: new for new, old in enumerate(order)}
    labels_sorted = np.array([label_map[l] for l in raw_labels])
    df['聚类标签'] = labels_sorted  # 0..k-1，按BMI从低到高排序

    # 计算BMI区间边界（中心点中点作为边界）
    centers_sorted_std = centers_std[order]
    if len(centers_sorted_std) > 1:
        boundaries_std = (centers_sorted_std[:-1] + centers_sorted_std[1:]) / 2
        boundaries_orig = bmi_scaler.inverse_transform(boundaries_std.reshape(-1, 1)).reshape(-1)
    else:
        boundaries_std = np.array([])
        boundaries_orig = np.array([])

    # 生成各组BMI区间字符串说明并写入（使用数据的最小/最大BMI作为端点，避免无穷）
    min_bmi, max_bmi = df['BMI'].min(), df['BMI'].max()
    bin_edges = np.concatenate(([min_bmi], boundaries_orig, [max_bmi]))
    interval_labels = []
    for i in range(optimal_k):
        left, right = bin_edges[i], bin_edges[i+1]
        if i < optimal_k - 1:
            interval_str = f"[{left:.1f}, {right:.1f})"
        else:
            interval_str = f"[{left:.1f}, {right:.1f}]"
        interval_labels.append(interval_str)
    df['BMI区间'] = [interval_labels[i] for i in df['聚类标签']]

    # 7. 分析聚类结果与最佳时点
    print("\n聚类结果分析（BMI一维分组）:")
    optimal_timing_results = []

    for i in range(optimal_k):
        cluster_data = df[df['聚类标签'] == i]
        print(f"\n第{i+1}组 (n={len(cluster_data)}):")
        print(f"  BMI区间: {interval_labels[i]}")
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
            'BMI区间': interval_labels[i],
            'BMI范围': f"{cluster_data['BMI'].min():.1f}-{cluster_data['BMI'].max():.1f}",
            'BMI均值': cluster_data['BMI'].mean(),
            '10%分位数': t_10,
            '20%分位数': t_20,
            '最佳NIPT时点': optimal_timing,
            '覆盖率': coverage_rate
        })

    # 8. 可视化聚类结果
    plt.figure(figsize=(12, 5))

    # 原始数据聚类图
    plt.subplot(1, 2, 1)
    colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))
    for i in range(optimal_k):
        cluster_data = df[df['聚类标签'] == i]
        plt.scatter(cluster_data['BMI'], cluster_data['最早达标孕周'],
                    c=[colors[i]], label=f'第{i+1}组', alpha=0.7)
    # 画出BMI区间边界
    for b in boundaries_orig:
        plt.axvline(b, color='k', linestyle='--', alpha=0.6)

    plt.xlabel('BMI')
    plt.ylabel('最早达标孕周')
    plt.title('聚类结果 (原始数据，BMI一维分组)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 标准化数据聚类图
    plt.subplot(1, 2, 2)
    for i in range(optimal_k):
        cluster_data = df[df['聚类标签'] == i]
        plt.scatter(cluster_data['BMI_标准化'], cluster_data['最早达标孕周_标准化'],
                    c=[colors[i]], label=f'第{i+1}组', alpha=0.7)
    # 标准化空间中的区间边界
    for b in boundaries_std:
        plt.axvline(b, color='k', linestyle='--', alpha=0.6)

    plt.xlabel('BMI (标准化)')
    plt.ylabel('最早达标孕周 (标准化)')
    plt.title('聚类结果 (标准化数据，BMI一维分组)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/Mac/Downloads/mm/2/results/clustering_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 9. 输出最佳NIPT时点汇总
    print("\n" + "="*60)
    print("最佳NIPT时点汇总")
    print("="*60)

    timing_summary_df = pd.DataFrame(optimal_timing_results)
    print(timing_summary_df.to_string(index=False, float_format='%.1f'))

    # 10. 保存结果
    output_file = '/Users/Mac/Downloads/mm/2/results/clustered_data.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n聚类结果已保存至: {output_file}")

    # 保存最佳时点汇总
    timing_file = '/Users/Mac/Downloads/mm/2/results/optimal_timing_summary.csv'
    timing_summary_df.to_csv(timing_file, index=False, encoding='utf-8-sig')
    print(f"最佳NIPT时点汇总已保存至: {timing_file}")

    # 返回与之前兼容：返回用于聚类的scaler（此处为bmi_scaler）
    return df, optimal_k, final_kmeans, bmi_scaler, timing_summary_df

if __name__ == "__main__":
    clustered_data, k, model, scaler, timing_summary = perform_clustering()
