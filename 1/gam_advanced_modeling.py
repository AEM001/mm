
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用广义可加模型 (GAM) 进行高级建模，探索交互项和更多变量
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, te
import os
import sys
import itertools

# 确保可以导入项目根目录的模块
sys.path.append('/Users/Mac/Downloads/mm')
from set_chinese_font import set_chinese_font

# 设置matplotlib以正确显示中文
set_chinese_font()

def gam_advanced_modeling():
    """
    使用高级GAM对Y染色体浓度进行建模，包括更多变量和交互项
    """
    
    # 1. 创建输出目录
    output_dir = '1/gam_advanced_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 2. 加载和准备数据
    print("加载和预处理数据...")
    try:
        original_df = pd.read_excel('/Users/Mac/Downloads/mm/附件.xlsx', sheet_name='男胎检测数据')
    except FileNotFoundError:
        print("错误: 未找到 '附件.xlsx'。")
        return

    # 执行与之前脚本相同的清洗和筛选流程
    ae_col = original_df.columns[-1]
    original_df = original_df[original_df[ae_col] == '是'].copy()
    quality_cols_indices = [11, 12, 13, 14, 15, 26]
    for col_idx in quality_cols_indices:
        col_data = pd.to_numeric(original_df.iloc[:, col_idx], errors='coerce')
        mean_val = col_data.mean()
        std_val = col_data.std()
        if col_idx in [11, 12, 14, 15]:
            threshold = mean_val - 3 * std_val
            original_df = original_df[col_data >= threshold].copy()
        else:
            threshold = mean_val + 3 * std_val
            original_df = original_df[col_data <= threshold].copy()

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
        except: return np.nan
    original_df['孕周_小数'] = original_df.iloc[:, 9].apply(convert_gestational_week)
    df_filtered = original_df[(original_df['孕周_小数'] >= 10) & (original_df['孕周_小数'] <= 25)].copy()

    # 提取所有需要的变量 (未标准化)
    y = df_filtered.iloc[:, 21].copy() # Y染色体浓度
    W = df_filtered['孕周_小数'].copy()
    B = pd.to_numeric(df_filtered.iloc[:, 10], errors='coerce').copy() # BMI
    A = pd.to_numeric(df_filtered.iloc[:, 2], errors='coerce').copy()  # 年龄
    
    def clean_pregnancy_count(value):
        if pd.isna(value): return np.nan
        value_str = str(value).strip()
        if '≥' in value_str or '>=' in value_str:
            import re
            numbers = re.findall(r'\d+', value_str)
            return float(numbers[0]) if numbers else np.nan
        try: return float(value_str)
        except: return np.nan
        
    C = df_filtered.iloc[:, 28].apply(clean_pregnancy_count) # 怀孕次数
    D = pd.to_numeric(df_filtered.iloc[:, 29], errors='coerce').copy() # 生产次数

    data = pd.DataFrame({'y': y, 'W': W, 'B': B, 'A': A, 'C': C, 'D': D}).dropna()
    print(f"移除缺失值后，最终有效样本数: {len(data)}")

    y = data['y']
    X = data[['W', 'B', 'A', 'C', 'D']]

    # 3. 构建和比较多个GAM模型
    print("\n开始构建和比较多个GAM模型...")
    models = {}
    
    # 模型1: 基准模型
    print("  - 正在训练模型1 (基准模型)...")
    models['M1_baseline'] = LinearGAM(s(0) + s(1) + s(2)).fit(X[['W', 'B', 'A']], y)

    # 模型2: 加入怀孕和生产次数
    print("  - 正在训练模型2 (加入C, D)...")
    models['M2_add_C_D'] = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4)).fit(X, y)

    # 模型3: 孕周与BMI交互
    print("  - 正在训练模型3 (W*B 交互)...")
    models['M3_interact_WB'] = LinearGAM(te(0, 1) + s(2)).fit(X[['W', 'B', 'A']], y)
    
    # 模型4: 孕周与年龄交互
    print("  - 正在训练模型4 (W*A 交互)...")
    models['M4_interact_WA'] = LinearGAM(te(0, 2) + s(1)).fit(X[['W', 'A', 'B']], y)

    # 模型5: 全交互模型 (W*B, W*A)
    print("  - 正在训练模型5 (全交互模型)...")
    # 注意: pygam中不能简单地用+连接多个te项，需要选择一个主交互
    # 这里我们选择一个更复杂的模型作为示例
    models['M5_full_interact'] = LinearGAM(te(0, 1) + te(0, 2) + s(3) + s(4)).fit(X, y)

    # 4. 比较模型性能并选择最优模型
    print("\n模型性能比较:")
    results = []
    for name, model in models.items():
        results.append({
            '模型': name,
            '伪R²': f"{model.statistics_['pseudo_r2']['explained_deviance']:.4f}",
            'AICc': f"{model.statistics_['AICc']:.2f}",
            'GCV': f"{model.statistics_['GCV']:.4f}",
            'EDoF': f"{model.statistics_['edof']:.2f}"
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # 根据AICc选择最优模型 (值越小越好)
    best_model_name = min(models, key=lambda k: models[k].statistics_['AICc'])
    best_model = models[best_model_name]
    print(f"\n最优模型 (基于AICc): {best_model_name}")

    # 5. 分析并可视化最优模型
    print(f"\n最优模型 ({best_model_name}) 摘要:")
    summary_str = best_model.summary()
    print(summary_str)

    # 保存结果到文件
    with open(f'{output_dir}/best_model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"最优GAM模型分析结果: {best_model_name}\n")
        f.write("="*70 + "\n")
        f.write("模型比较:\n")
        f.write(results_df.to_string(index=False) + "\n\n")
        f.write("="*70 + "\n")
        f.write(f"最优模型 ({best_model_name}) 摘要:\n")
        f.write(str(summary_str))

    # 可视化偏依赖图
    print("\n为最优模型生成偏依赖图...")
    
    # Get non-intercept terms
    non_intercept_terms = [t for t in best_model.terms if not t.isintercept]
    n_terms = len(non_intercept_terms)

    fig, axes = plt.subplots(1, n_terms, figsize=(6 * n_terms, 5))
    if n_terms == 1: axes = [axes] # 确保axes是可迭代的
    
    term_titles = []
    if "interact" in best_model_name:
        if "WB" in best_model_name: term_titles.extend(['孕周(W) x BMI(B)', '年龄(A)'])
        elif "WA" in best_model_name: term_titles.extend(['孕周(W) x 年龄(A)', 'BMI(B)'])
        else: term_titles = [f'Term {i+1}' for i in range(n_terms)] # 备用标题
    else:
        term_titles = ['孕周(W)', 'BMI(B)', '年龄(A)', '怀孕次数(C)', '生产次数(D)']

    for i, term in enumerate(non_intercept_terms):
        ax = axes[i]
        XX = best_model.generate_X_grid(term=i)
        pdep, confi = best_model.partial_dependence(term=i, X=XX, width=0.95)
        
        if term.istensor:
            # 绘制交互项的热力图
            im = ax.imshow(pdep.reshape(100, 100).T, cmap='viridis', origin='lower',
                           extent=[XX[:, term.info['terms'][0]['feature']].min(), 
                                   XX[:, term.info['terms'][0]['feature']].max(),
                                   XX[:, term.info['terms'][1]['feature']].min(),
                                   XX[:, term.info['terms'][1]['feature']].max()],
                           aspect='auto')
            ax.set_xlabel(f"特征 {term.info['terms'][0]['feature']}")
            ax.set_ylabel(f"特征 {term.info['terms'][1]['feature']}")
            fig.colorbar(im, ax=ax)
        else:
            # 绘制平滑项的曲线图
            feature_index = term.feature
            ax.plot(XX[:, feature_index], pdep, color='blue')
            ax.plot(XX[:, feature_index], confi, color='grey', linestyle='--')
            ax.set_xlabel(term_titles[i])
            ax.set_ylabel('对Y染色体浓度的影响')
        
        ax.set_title(f'偏依赖图: {term_titles[i]}')
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(f'最优模型 ({best_model_name}) 偏依赖关系图', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = f'{output_dir}/best_model_partial_dependence.png'
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"\n分析完成。最优模型的结果已保存至 '{output_dir}' 目录。")

if __name__ == "__main__":
    gam_advanced_modeling()
