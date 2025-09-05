#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y染色体浓度与标准化变量的相关性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import sys
import warnings
warnings.filterwarnings('ignore')

# 确保可以导入项目根目录的模块
sys.path.append('/Users/Mac/Downloads/mm')
from set_chinese_font import set_chinese_font
set_chinese_font()

def correlation_analysis():
    """执行Y染色体浓度与标准化变量的相关性分析"""
    
    print("开始相关性分析...")
    
    # 1. 读取处理后的数据
    print("1. 读取数据...")
    try:
        df = pd.read_csv('/Users/Mac/Downloads/mm/1/processed_data.csv')
        print(f"数据量: {len(df)} 条记录")
    except FileNotFoundError:
        print("错误: 未找到processed_data.csv文件")
        return None
    
    # 2. 检查所需列是否存在
    required_columns = ['Y染色体浓度', '孕周_标准化', 'BMI_标准化', '年龄_标准化', '怀孕次数_标准化', '生产次数_标准化']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"错误: 缺少以下列: {missing_columns}")
        print(f"可用列: {list(df.columns)}")
        return None
    
    # 3. 提取分析变量
    print("2. 提取分析变量...")
    analysis_data = df[required_columns].copy()
    
    # 移除缺失值
    before_na = len(analysis_data)
    analysis_data = analysis_data.dropna()
    print(f"移除缺失值: {before_na - len(analysis_data)} 条，剩余: {len(analysis_data)} 条")
    
    if len(analysis_data) == 0:
        print("错误: 没有有效数据进行分析")
        return None
    
    # 4. 计算Spearman相关系数
    print("3. 计算Spearman相关系数...")
    
    y_concentration = analysis_data['Y染色体浓度']
    standardized_vars = ['孕周_标准化', 'BMI_标准化', '年龄_标准化', '怀孕次数_标准化', '生产次数_标准化']
    
    correlation_results = []
    
    for var in standardized_vars:
        corr_coef, p_value = spearmanr(y_concentration, analysis_data[var])
        
        # 判断显著性
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = ""
        
        # 判断相关性强度
        abs_corr = abs(corr_coef)
        if abs_corr >= 0.7:
            strength = "强"
        elif abs_corr >= 0.5:
            strength = "中等"
        elif abs_corr >= 0.3:
            strength = "弱"
        else:
            strength = "很弱"
        
        correlation_results.append({
            '变量': var.replace('_标准化', ''),
            'Spearman系数': corr_coef,
            'P值': p_value,
            '显著性': significance,
            '相关性强度': strength
        })
    
    # 5. 创建相关性结果表
    correlation_df = pd.DataFrame(correlation_results)
    
    print("\n相关性分析结果:")
    print("="*60)
    print(correlation_df.to_string(index=False, float_format='%.4f'))
    
    # 6. 可视化分析
    print("\n4. 生成可视化图表...")
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 为每个标准化变量创建散点图
    for i, var in enumerate(standardized_vars):
        ax = axes[i]
        
        # 散点图
        ax.scatter(analysis_data[var], y_concentration, alpha=0.6, s=30)
        
        # 添加趋势线
        z = np.polyfit(analysis_data[var], y_concentration, 1)
        p = np.poly1d(z)
        ax.plot(analysis_data[var], p(analysis_data[var]), "r--", alpha=0.8, linewidth=2)
        
        # 设置标题和标签
        corr_info = correlation_results[i]
        title = f"{corr_info['变量']} vs Y染色体浓度\n"
        title += f"Spearman: {corr_info['Spearman系数']:.3f}{corr_info['显著性']}"
        title += f" ({corr_info['相关性强度']}相关)"
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"{corr_info['变量']}(标准化)", fontsize=10)
        ax.set_ylabel('Y染色体浓度', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    if len(standardized_vars) < len(axes):
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('/Users/Mac/Downloads/mm/1/correlation_analysis/correlation_scatter_plots.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. 创建相关性热力图
    print("5. 生成相关性热力图...")
    
    # 计算完整的相关性矩阵
    correlation_matrix = analysis_data[required_columns].corr(method='spearman')
    
    plt.figure(figsize=(10, 8))
    
    # 创建热力图
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8})
    
    plt.title('Y染色体浓度与标准化变量Spearman相关性矩阵', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('/Users/Mac/Downloads/mm/1/correlation_analysis/correlation_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. 生成详细统计表
    print("6. 生成详细统计表...")
    
    # 描述性统计
    descriptive_stats = analysis_data.describe()
    
    # 保存结果
    correlation_df.to_csv('/Users/Mac/Downloads/mm/1/correlation_analysis/spearman_correlation_results.csv', 
                         index=False, encoding='utf-8-sig')
    
    descriptive_stats.to_csv('/Users/Mac/Downloads/mm/1/correlation_analysis/descriptive_statistics.csv', 
                            encoding='utf-8-sig')
    
    correlation_matrix.to_csv('/Users/Mac/Downloads/mm/1/correlation_analysis/full_correlation_matrix.csv', 
                             encoding='utf-8-sig')
    
    # 9. 生成分析报告摘要
    print("7. 生成分析报告...")
    
    # 找出最强和最弱的相关性
    abs_correlations = [(abs(r['Spearman系数']), r) for r in correlation_results]
    abs_correlations.sort(reverse=True)
    
    strongest_corr = abs_correlations[0][1]
    weakest_corr = abs_correlations[-1][1]
    
    # 统计显著相关的变量数量
    significant_vars = [r for r in correlation_results if r['P值'] < 0.05]
    
    report_summary = f"""
相关性分析报告摘要
{'='*50}

数据概况:
- 分析样本数: {len(analysis_data)}
- 分析变量数: {len(standardized_vars)}

主要发现:
- 最强相关变量: {strongest_corr['变量']} (r={strongest_corr['Spearman系数']:.3f}, {strongest_corr['相关性强度']}相关)
- 最弱相关变量: {weakest_corr['变量']} (r={weakest_corr['Spearman系数']:.3f}, {weakest_corr['相关性强度']}相关)
- 显著相关变量数: {len(significant_vars)}/{len(standardized_vars)}

显著性说明:
*** p<0.001, ** p<0.01, * p<0.05

输出文件:
- spearman_correlation_results.csv: Spearman相关系数表
- correlation_scatter_plots.png: 散点图矩阵
- correlation_heatmap.png: 相关性热力图
- full_correlation_matrix.csv: 完整相关性矩阵
- descriptive_statistics.csv: 描述性统计
"""
    
    print(report_summary)
    
    # 保存报告摘要
    with open('/Users/Mac/Downloads/mm/1/correlation_analysis/analysis_report_summary.txt', 'w', encoding='utf-8') as f:
        f.write(report_summary)
    
    print(f"\n所有结果已保存至: /Users/Mac/Downloads/mm/1/correlation_analysis/")
    
    return correlation_df, correlation_matrix, analysis_data

if __name__ == "__main__":
    results = correlation_analysis()
