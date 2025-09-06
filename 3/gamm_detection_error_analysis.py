#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题三（GAMM+KMeans）检测误差影响分析
- 基于 `3/gamm_y_chromosome_prediction.py` 的 R/mgcv GAMM via rpy2 预测
- 基于 `3/kmeans_bmi_segmentation.py` 的 KMeans 分群
- 复用 `2/detection_error_analysis.py` 的误差分析模块

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

# 导入问题二的检测误差分析工具（向后兼容）
TOOLS_DIR_Q2 = os.path.join(PROJECT_ROOT, '2')
if TOOLS_DIR_Q2 not in sys.path:
    sys.path.insert(0, TOOLS_DIR_Q2)
from detection_error_analysis import (
    DetectionErrorAnalyzer,
    create_error_visualizations,
    generate_error_report,
    run_error_analysis_with_strategy,  # 新增：支持策略模式
)


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

    # 强制检查 R/mgcv 可用性（满足项目约束：仅用 R/mgcv GAMM）
    try:
        if not getattr(bmi_analyzer.gamm_predictor, 'use_r_gamm', False):
            raise RuntimeError(
                "R/mgcv GAMM 不可用。请确保已安装 rpy2，并在 R 中安装 mgcv 包。"
            )
    except AttributeError:
        # 如果对象不含该属性，直接抛错提示环境
        raise RuntimeError("无法确认 R/mgcv GAMM 可用性，请检查 rpy2 与 mgcv 环境。")

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
    analyzer = DetectionErrorAnalyzer(output_dir=output_dir)

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
