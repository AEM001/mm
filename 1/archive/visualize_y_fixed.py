#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y染色体浓度分布及其与关键变量关系的可视化分析 - 修复版
"""

import pandas as pd
import numpy as np
import sys
import os

# 设置matplotlib后端和中文字体 - 必须在导入pyplot之前
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 导入matplotlib字体管理器
import matplotlib.font_manager as fm

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy import stats

# 强制设置中文字体 - 更直接的方法
def force_chinese_font():
    """强制设置中文字体"""
    # 直接设置字体路径
    font_paths = [
        '/System/Library/Fonts/Songti.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc', 
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/Library/Fonts/Songti.ttc'
    ]
    
    # 尝试使用系统字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = ['Songti SC', 'STHeiti', 'Heiti TC', 'STSong', 'SimSong', 'Kaiti SC']
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.family'] = 'sans-serif'
            print(f"成功设置中文字体: {font}")
            
            # 测试字体
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试中文', fontsize=12, ha='center')
            plt.close(fig)
            return True
    
    print("警告: 无法设置中文字体，将使用英文标签")
    return False

# 设置中文字体
chinese_ok = force_chinese_font()

def visualize_y_distribution():
    """可视化Y染色体浓度分布及其与关键变量的关系"""
    
    # 创建输出目录
    output_dir = 'y_distribution_analysis'
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 读取处理后的数据
    print("读取数据...")
    df = pd.read_csv('processed_data.csv')
    print(f"数据样本数: {len(df)}")
    
    # 关键变量列表
    key_vars = ['孕周_标准化', 'BMI_标准化', '年龄_标准化', '怀孕次数_标准化', '生产次数_标准化']
    y_var = 'Y染色体浓度'
    
    # 如果中文字体不可用，使用英文标签
    if not chinese_ok:
        var_labels = {
            '孕周_标准化': 'Gestational Week (Std)',
            'BMI_标准化': 'BMI (Std)', 
            '年龄_标准化': 'Age (Std)',
            '怀孕次数_标准化': 'Pregnancy Count (Std)',
            '生产次数_标准化': 'Birth Count (Std)',
            'Y染色体浓度': 'Y Chromosome Concentration'
        }
        plot_titles = {
            'Y染色体浓度分布 (logit变换后)': 'Y Chromosome Concentration Distribution (logit transformed)',
            'Q-Q图 (正态性检验)': 'Q-Q Plot (Normality Test)',
            'Y染色体浓度箱线图': 'Y Chromosome Concentration Box Plot',
            '变量相关系数热力图': 'Variable Correlation Heatmap'
        }
    else:
        var_labels = {var: var for var in key_vars + [y_var]}
        plot_titles = {
            'Y染色体浓度分布 (logit变换后)': 'Y染色体浓度分布 (logit变换后)',
            'Q-Q图 (正态性检验)': 'Q-Q图 (正态性检验)',
            'Y染色体浓度箱线图': 'Y染色体浓度箱线图',
            '变量相关系数热力图': '变量相关系数热力图'
        }
    
    # 创建图形布局 - 3行3列
    fig = plt.figure(figsize=(18, 15))
    
    # 1. Y染色体浓度分布图 (左上)
    ax1 = plt.subplot(3, 3, 1)
    
    # 直方图 + 密度曲线
    n, bins, patches = plt.hist(df[y_var], bins=30, density=True, alpha=0.7, 
                               color='skyblue', edgecolor='black', linewidth=0.5)
    
    # 添加核密度估计曲线
    kde_x = np.linspace(df[y_var].min(), df[y_var].max(), 100)
    kde = stats.gaussian_kde(df[y_var].dropna())
    plt.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE' if not chinese_ok else '核密度估计')
    
    # 添加正态分布拟合曲线
    mu, sigma = stats.norm.fit(df[y_var].dropna())
    normal_curve = stats.norm.pdf(kde_x, mu, sigma)
    plt.plot(kde_x, normal_curve, 'g--', linewidth=2, 
             label=f'Normal Fit (μ={mu:.3f}, σ={sigma:.3f})' if not chinese_ok else f'正态拟合 (μ={mu:.3f}, σ={sigma:.3f})')
    
    plt.xlabel(var_labels[y_var])
    plt.ylabel('Density' if not chinese_ok else '密度')
    plt.title(plot_titles['Y染色体浓度分布 (logit变换后)'], fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'N: {len(df[y_var].dropna())}\n'
    stats_text += f'Mean: {df[y_var].mean():.3f}\n' if not chinese_ok else f'均值: {df[y_var].mean():.3f}\n'
    stats_text += f'Std: {df[y_var].std():.3f}\n' if not chinese_ok else f'标准差: {df[y_var].std():.3f}\n'
    stats_text += f'Skew: {stats.skew(df[y_var].dropna()):.3f}\n' if not chinese_ok else f'偏度: {stats.skew(df[y_var].dropna()):.3f}\n'
    stats_text += f'Kurt: {stats.kurtosis(df[y_var].dropna()):.3f}' if not chinese_ok else f'峰度: {stats.kurtosis(df[y_var].dropna()):.3f}'
    
    plt.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Q-Q图检验正态性 (右上)
    ax2 = plt.subplot(3, 3, 2)
    stats.probplot(df[y_var].dropna(), dist="norm", plot=plt)
    plt.title(plot_titles['Q-Q图 (正态性检验)'], fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. 箱线图 (中上)
    ax3 = plt.subplot(3, 3, 3)
    box_plot = plt.boxplot(df[y_var].dropna(), patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightcoral')
    plt.ylabel(var_labels[y_var])
    plt.title(plot_titles['Y染色体浓度箱线图'], fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4-8. Y与各关键变量的散点图
    positions = [4, 5, 6, 7, 8]  # 对应subplot位置
    
    for i, var in enumerate(key_vars):
        ax = plt.subplot(3, 3, positions[i])
        
        # 散点图
        plt.scatter(df[var], df[y_var], alpha=0.6, s=20, color='steelblue')
        
        # 添加回归线
        valid_data = df[[var, y_var]].dropna()
        if len(valid_data) > 1:
            z = np.polyfit(valid_data[var], valid_data[y_var], 1)
            p = np.poly1d(z)
            plt.plot(valid_data[var].sort_values(), p(valid_data[var].sort_values()), 
                    "r--", alpha=0.8, linewidth=2)
            
            # 计算相关系数
            corr_coef = valid_data[var].corr(valid_data[y_var])
            
            # 添加相关系数文本
            corr_text = f'r: {corr_coef:.3f}' if not chinese_ok else f'相关系数: {corr_coef:.3f}'
            plt.text(0.05, 0.95, corr_text, 
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.xlabel(var_labels[var])
        plt.ylabel(var_labels[y_var])
        title_text = f'{var_labels[y_var]} vs {var_labels[var]}' if not chinese_ok else f'Y染色体浓度 vs {var}'
        plt.title(title_text, fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    # 9. 相关系数热力图 (右下)
    ax9 = plt.subplot(3, 3, 9)
    
    # 计算相关矩阵
    corr_vars = [y_var] + key_vars
    corr_matrix = df[corr_vars].corr()
    
    # 如果不支持中文，重命名列和索引
    if not chinese_ok:
        corr_matrix.index = [var_labels[var] for var in corr_matrix.index]
        corr_matrix.columns = [var_labels[var] for var in corr_matrix.columns]
    
    # 绘制热力图
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.3f')
    plt.title(plot_titles['变量相关系数热力图'], fontsize=12, fontweight='bold')
    
    # 调整布局
    plt.tight_layout(pad=2.0)
    
    # 保存图形到指定目录
    output_file = os.path.join(output_dir, 'y_distribution_analysis_fixed.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"可视化结果已保存至: {output_file}")
    
    plt.close()  # 关闭图形
    
    # 输出统计摘要并保存到文件
    summary_text = []
    summary_text.append("="*60)
    summary_text.append("Y染色体浓度统计摘要 (logit变换后)")
    summary_text.append("="*60)
    summary_text.append(f"样本数量: {len(df[y_var].dropna())}")
    summary_text.append(f"均值: {df[y_var].mean():.4f}")
    summary_text.append(f"标准差: {df[y_var].std():.4f}")
    summary_text.append(f"最小值: {df[y_var].min():.4f}")
    summary_text.append(f"最大值: {df[y_var].max():.4f}")
    summary_text.append(f"中位数: {df[y_var].median():.4f}")
    summary_text.append(f"偏度: {stats.skew(df[y_var].dropna()):.4f}")
    summary_text.append(f"峰度: {stats.kurtosis(df[y_var].dropna()):.4f}")
    
    # 正态性检验
    shapiro_stat, shapiro_p = stats.shapiro(df[y_var].dropna()[:5000])  # shapiro最多支持5000样本
    summary_text.append(f"\nShapiro-Wilk正态性检验:")
    summary_text.append(f"统计量: {shapiro_stat:.4f}, p值: {shapiro_p:.2e}")
    summary_text.append(f"结论: {'数据符合正态分布' if shapiro_p > 0.05 else '数据不符合正态分布'}")
    
    summary_text.append(f"\n与关键变量的相关系数:")
    summary_text.append("-" * 40)
    for var in key_vars:
        corr_coef = df[var].corr(df[y_var])
        summary_text.append(f"{var:15s}: {corr_coef:7.4f}")
    
    # 保存相关系数矩阵到CSV文件
    corr_file = os.path.join(output_dir, 'correlation_matrix_fixed.csv')
    df[corr_vars].corr().to_csv(corr_file, encoding='utf-8-sig')
    summary_text.append(f"\n相关系数矩阵已保存至: {corr_file}")
    
    # 保存统计摘要到文件
    summary_file = os.path.join(output_dir, 'statistical_summary_fixed.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_text))
    
    # 输出到控制台
    print("\n" + '\n'.join(summary_text))
    print(f"\n统计摘要已保存至: {summary_file}")
    print(f"\n所有分析结果已保存至目录: {output_dir}")
    print("分析完成！")

if __name__ == "__main__":
    visualize_y_distribution()
