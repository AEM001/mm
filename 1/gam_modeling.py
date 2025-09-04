#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用广义可加模型 (GAM) 分析Y染色体浓度的影响因素
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, te
import os
import sys

# 确保可以导入项目根目录的模块
sys.path.append('/Users/Mac/Downloads/mm')
from set_chinese_font import set_chinese_font

# 设置matplotlib以正确显示中文
set_chinese_font()

def gam_modeling():
    """
    使用GAM对Y染色体浓度进行建模分析
    """
    
    # 1. 创建输出目录
    output_dir = '1/gam_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 2. 加载和准备数据 (与regression_modeling.py保持一致)
    print("加载和预处理数据...")
    
    # 读取原始数据以获取未标准化的变量
    try:
        original_df = pd.read_excel('/Users/Mac/Downloads/mm/附件.xlsx', sheet_name='男胎检测数据')
    except FileNotFoundError:
        print("错误: 未找到 '附件.xlsx'。请确保文件位于 /Users/Mac/Downloads/mm/ 目录下。")
        return

    # 数据清洗 - 与data_process.py保持一致
    ae_col = original_df.columns[-1]
    original_df = original_df[original_df[ae_col] == '是'].copy()
    
    # 质量控制筛选
    quality_cols_indices = [11, 12, 13, 14, 15, 26] # L, M, N, O, P, AA
    for col_idx in quality_cols_indices:
        col_data = pd.to_numeric(original_df.iloc[:, col_idx], errors='coerce')
        mean_val = col_data.mean()
        std_val = col_data.std()
        if col_idx in [11, 12, 14, 15]: # 剔除低值
            threshold = mean_val - 3 * std_val
            original_df = original_df[col_data >= threshold].copy()
        else: # 剔除高值
            threshold = mean_val + 3 * std_val
            original_df = original_df[col_data <= threshold].copy()

    # 转换孕周格式
    def convert_gestational_week(week_str):
        if pd.isna(week_str): return np.nan
        try:
            week_str = str(week_str).strip()
            if 'w' in week_str:
                parts = week_str.split('w')
                weeks = float(parts[0])
                days = float(parts[1].replace('+', '')) if '+' in parts[1] else 0
                return weeks + days / 7.0
            return float(week_str)
        except:
            return np.nan
            
    original_df['孕周_小数'] = original_df.iloc[:, 9].apply(convert_gestational_week)
    
    # 筛选10-25周的样本
    df_filtered = original_df[(original_df['孕周_小数'] >= 10) & (original_df['孕周_小数'] <= 25)].copy()
    print(f"筛选10-25周样本后，剩余: {len(df_filtered)} 个")

    # 提取因变量和自变量 (使用未标准化的原始值)
    y = df_filtered.iloc[:, 21].copy() # V列 - Y染色体浓度
    W = df_filtered['孕周_小数'].copy()
    B = pd.to_numeric(df_filtered.iloc[:, 10], errors='coerce').copy() # K列 - BMI
    A = pd.to_numeric(df_filtered.iloc[:, 2], errors='coerce').copy()  # C列 - 年龄

    # 合并为新的DataFrame并移除缺失值
    data = pd.DataFrame({'y': y, 'W': W, 'B': B, 'A': A}).dropna()
    print(f"移除缺失值后，最终用于建模的有效样本数: {len(data)}")

    y = data['y']
    X = data[['W', 'B', 'A']]

    # 3. 构建并训练GAM模型
    print("构建并训练GAM模型...")
    
    # 模型包含三个自变量的平滑项
    # s(0) -> W (孕周), s(1) -> B (BMI), s(2) -> A (年龄)
    gam = LinearGAM(s(0) + s(1) + s(2)).fit(X, y)
    
    # 4. 输出模型摘要
    print("GAM模型摘要:")
    summary_str = gam.summary()
    print(summary_str)
    
    # 保存摘要到文件
    with open(f'{output_dir}/gam_summary.txt', 'w', encoding='utf-8') as f:
        f.write("广义可加模型(GAM)分析结果")
        f.write("="*50 + "")
        f.write(f"因变量: Y染色体浓度")
        f.write(f"自变量: 孕周 (W), BMI (B), 年龄 (A)")
        f.write(f"有效样本数: {len(data)}")
        f.write("="*50 + "")
        f.write(str(summary_str))

    # 5. 可视化偏依赖图
    print("生成偏依赖图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ['孕周 (W)', 'BMI (B)', '年龄 (A)']
    
    for i, ax in enumerate(axes):
        # partial_dependence为每个平滑项生成数据点和置信区间
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        
        ax.plot(XX[:, i], pdep, color='blue')
        ax.plot(XX[:, i], confi, color='grey', linestyle='--')
        ax.set_title(f'偏依赖图: {titles[i]}')
        ax.set_xlabel(titles[i])
        ax.set_ylabel('对Y染色体浓度的影响 (样条函数)')
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle('GAM模型各变量的偏依赖关系图', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图像
    plot_path = f'{output_dir}/partial_dependence_plots.png'
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    print(f"偏依赖图已保存至: {plot_path}")
    print(f"模型摘要已保存至: {output_dir}/gam_summary.txt")
    print("GAM分析完成。请查看 '1/gam_results' 文件夹中的结果。")

if __name__ == "__main__":
    gam_modeling()
