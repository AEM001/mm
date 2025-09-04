#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y染色体浓度与孕周、BMI的相关性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
sys.path.append('/Users/Mac/Downloads/mm')
from set_chinese_font import set_chinese_font

# 设置中文字体
set_chinese_font()

def correlation_analysis():
    """分析Y染色体浓度与孕周、BMI的相关性"""
    
    # 创建输出目录
    output_dir = 'correlation_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取处理后的数据
    df = pd.read_csv('processed_data.csv')
    
    # 提取需要分析的变量
    y_concentration = df['Y染色体浓度']
    variables = {
        '孕周': df['孕周_标准化'],
        'BMI': df['BMI_标准化'], 
        '年龄': df['年龄_标准化'],
        '怀孕次数': df['怀孕次数_标准化'],
        '生产次数': df['生产次数_标准化']
    }
    
    # 绘制散点图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (var_name, var_data) in enumerate(variables.items()):
        axes[i].scatter(var_data, y_concentration, alpha=0.6, s=20)
        axes[i].set_xlabel(f'{var_name} (标准化)')
        axes[i].set_ylabel('Y染色体浓度')
        axes[i].set_title(f'Y染色体浓度 vs {var_name}')
        axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 计算相关系数
    def get_correlation_strength(r):
        if abs(r) >= 0.7:
            return "强"
        elif abs(r) >= 0.3:
            return "中等"
        else:
            return "弱"
    
    def get_direction(r):
        return "正" if r > 0 else "负"
    
    def get_significance(p):
        return "显著" if p < 0.05 else "不显著"
    
    # 计算所有变量的相关系数
    results = []
    for var_name, var_data in variables.items():
        # 移除缺失值
        valid_mask = ~(var_data.isna() | y_concentration.isna())
        
        # Pearson相关系数
        pearson_r, pearson_p = stats.pearsonr(
            var_data[valid_mask], 
            y_concentration[valid_mask]
        )
        
        # Spearman相关系数
        spearman_r, spearman_p = stats.spearmanr(
            var_data[valid_mask], 
            y_concentration[valid_mask]
        )
        
        results.append({
            '指标': var_name,
            'Pearson相关系数': f"{pearson_r:.4f}",
            'Pearson p值': f"{pearson_p:.6f}" if pearson_p >= 0.000001 else f"{pearson_p:.2e}",
            'Spearman相关系数': f"{spearman_r:.4f}",
            'Spearman p值': f"{spearman_p:.6f}" if spearman_p >= 0.000001 else f"{spearman_p:.2e}",
            '关联方向': get_direction(spearman_r),
            '关联强度': get_correlation_strength(spearman_r),
            '显著性': get_significance(spearman_p)
        })
    
    results_df = pd.DataFrame(results)
    
    print("\nY染色体浓度相关性分析结果")
    print("=" * 100)
    print(results_df.to_string(index=False))
    
    # 保存结果
    results_df.to_csv(f'{output_dir}/correlation_results.csv', index=False, encoding='utf-8-sig')
    
    return results_df

if __name__ == "__main__":
    correlation_analysis()
