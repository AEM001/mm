#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用广义可加模型 (GAM) 进行高级建模，探索变量之间的非线性关系和交互效应
增强版：添加了可视化、诊断和详细表格输出
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, te, l
import os
import sys
import itertools
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.api as sm
from scipy import stats

# 确保可以导入项目根目录的模块
sys.path.append('/Users/Mac/Downloads/mm')
from set_chinese_font import set_chinese_font

# 设置matplotlib以正确显示中文
set_chinese_font()

def gam_enhanced_modeling():
    """
    使用高级GAM对Y染色体浓度进行建模，包括更多变量和交互项
    增强版：添加偏依赖图、模型诊断和详细结果表格
    """
    
    # 1. 创建输出目录
    output_dir = 'gam_enhanced_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 2. 加载已处理的数据
    print("加载已处理的数据...")
    try:
        data = pd.read_csv('processed_data.csv')
    except FileNotFoundError:
        print("错误: 未找到 'processed_data.csv'。")
        return

    print(f"加载的数据样本数: {len(data)}")
    
    # 提取建模所需的变量（使用标准化后的数据）
    y = data['Y染色体浓度'].copy()  # 目标变量
    X = data[['孕周_标准化', 'BMI_标准化', '年龄_标准化', '怀孕次数_标准化', '生产次数_标准化']].copy()
    
    # 保留原始列名，不再重命名为W, B, A等缩写
    feature_names = X.columns.tolist()
    
    print(f"清理后的建模数据样本数: {len(y)}")

    # 3. 构建和比较多个GAM模型
    print("\n开始构建和比较多个GAM模型...")
    models = {}
    
    # 模型1: 基准模型（孕周、BMI和年龄都用平滑项）
    print("  - 正在训练模型1 (基准模型)...")
    models['M1_baseline'] = LinearGAM(s(0) + s(1) + s(2)).fit(X[feature_names[:3]], y)

    # 模型2: 加入怀孕和生产次数（作为线性项而非平滑项）
    print("  - 正在训练模型2 (加入怀孕次数和生产次数为线性项)...")
    models['M2_add_C_D'] = LinearGAM(s(0) + s(1) + s(2) + l(3) + l(4)).fit(X, y)

    # 模型3: 孕周与BMI交互
    print("  - 正在训练模型3 (孕周×BMI交互)...")
    models['M3_interact_WB'] = LinearGAM(te(0, 1) + s(2)).fit(X[feature_names[:3]], y)
    
    # 模型4: 孕周与年龄交互
    print("  - 正在训练模型4 (孕周×年龄交互)...")
    models['M4_interact_WA'] = LinearGAM(te(0, 2) + s(1)).fit(X[feature_names[:3]], y)

    # 模型5: 全交互模型 (孕周×BMI, 孕周×年龄, 以及怀孕/生产次数作为线性项)
    print("  - 正在训练模型5 (全交互模型)...")
    models['M5_full_interact'] = LinearGAM(te(0, 1) + te(0, 2) + l(3) + l(4)).fit(X, y)

    # 4. 比较模型性能并选择最优模型
    print("\n模型性能比较:")
    results = []
    for name, model in models.items():
        # 计算显著性检验
        p_values = model.statistics_['p_values']
        significant_terms = sum(p < 0.05 for p in p_values)
        
        results.append({
            '模型': name,
            '伪R²': f"{model.statistics_['pseudo_r2']['explained_deviance']:.4f}",
            'AICc': f"{model.statistics_['AICc']:.2f}",
            'GCV': f"{model.statistics_['GCV']:.4f}",
            'EDoF': f"{model.statistics_['edof']:.2f}",
            '显著项数': f"{significant_terms}/{len(p_values)}"
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
    
    # 确定最佳模型使用了哪些特征
    # 这里需要根据模型名称动态判断
    if best_model_name in ['M1_baseline', 'M3_interact_WB', 'M4_interact_WA']:
        best_model_features = feature_names[:3]  # 只有前三个特征：孕周、BMI、年龄
    else:
        best_model_features = feature_names  # 全部五个特征
    
    # 创建详细的模型摘要表格
    model_details = create_detailed_summary(best_model, best_model_features)
    print("\n详细模型摘要表格:")
    print(model_details.to_string())
    
    # 保存详细模型摘要表格
    model_details.to_csv(f'{output_dir}/detailed_model_summary.csv', index=False, encoding='utf-8-sig')
    
    # 6. 生成偏依赖图
    print("\n生成偏依赖图...")
    create_partial_dependence_plots(best_model, best_model_features, output_dir)
    
    # 7. 进行模型诊断
    print("\n进行模型诊断...")
    perform_model_diagnostics(best_model, X, y, best_model_features, output_dir)
    
    # 保存基本结果到文本文件
    with open(f'{output_dir}/best_model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"最优GAM模型分析结果: {best_model_name}\n")
        f.write("="*70 + "\n")
        f.write("模型比较:\n")
        f.write(results_df.to_string(index=False) + "\n\n")
        f.write("="*70 + "\n")
        f.write(f"最优模型 ({best_model_name}) 摘要:\n")
        f.write(str(summary_str) + "\n\n")
        
    print(f"\n分析完成。所有结果已保存至 '{output_dir}' 目录。")

def create_detailed_summary(model, feature_names):
    """创建详细的模型摘要表格"""
    
    rows = []
    
    # 分析模型中的每个项
    for i, term in enumerate(model.terms):
        term_type = term.__class__.__name__  # 获取项的类型
        
        # 创建可读的项名称
        if term_type == 'SplineTerm':
            # 平滑项
            term_idx = term.feature
            term_name = f"s({feature_names[term_idx]})"
            term_type_cn = "平滑项 (Smooth)"
        elif term_type == 'LinearTerm':
            # 线性项
            term_idx = term.feature
            term_name = f"l({feature_names[term_idx]})"
            term_type_cn = "线性项 (Linear)"
        elif term_type == 'TensorTerm':
            # 交互项 - 从 info 中获取特征信息
            try:
                # 从 tensor term 的 info 中提取特征索引
                terms_info = term.info['terms']
                feat1 = int(terms_info[0]['feature'])
                feat2 = int(terms_info[1]['feature']) 
            except (KeyError, IndexError, ValueError):
                # 如果无法获取特征索引，使用默认值
                feat1, feat2 = 0, 1
            
            feat_name1 = feature_names[feat1] if feat1 < len(feature_names) else f'feature_{feat1}'
            feat_name2 = feature_names[feat2] if feat2 < len(feature_names) else f'feature_{feat2}'
            term_name = f"te({feat_name1}, {feat_name2})"
            term_type_cn = "交互项 (Tensor)"
        else:
            # 截距项或其他
            term_name = "截距 (Intercept)"
            term_type_cn = "常数项 (Constant)"
        
        # 获取统计信息
        p_value = model.statistics_['p_values'][i] if i < len(model.statistics_['p_values']) else np.nan
        
        # 确定显著性星号
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"
        
        # 获取EDoF (有效自由度) - 使用可用的统计信息
        edof = np.nan  # 暂时不提供单个项的EDoF
        
        # 获取Lambda (平滑惩罚系数) - 使用新的方法
        try:
            if hasattr(model, 'lam') and i < len(model.lam):
                lam_val = model.lam[i]
                # 处理数组情况
                if isinstance(lam_val, (list, tuple, np.ndarray)):
                    lam = lam_val[0] if len(lam_val) > 0 else np.nan
                else:
                    lam = lam_val
            else:
                lam = np.nan
        except (IndexError, TypeError):
            lam = np.nan
        
        # 添加行数据
        rows.append({
            'Term (模型项)': term_name,
            'Term_Type (项类型)': term_type_cn,
            'EDoF (有效自由度)': f"{edof:.3f}" if not np.isnan(edof) else "",
            'Lambda (平滑惩罚系数)': f"{lam:.6f}" if isinstance(lam, (int, float)) and not np.isnan(lam) else str(lam),
            'P-value (P值)': f"{p_value:.4f}" if not np.isnan(p_value) else "",
            'Significance (显著性水平)': significance,
            'Interpretation (业务解读)': "" # 留空，用于后续手动填写
        })
    
    # 创建DataFrame
    return pd.DataFrame(rows)

def create_partial_dependence_plots(model, feature_names, output_dir):
    """为模型中的每个项创建偏依赖图"""
    
    # 图表整体设置
    plt.figure(figsize=(18, 12))
    
    # 为每个模型项生成子图位置
    num_plots = len(model.terms)
    rows = max(1, (num_plots + 1) // 2)  # 向上取整
    cols = min(2, num_plots)  # 最多2列
    
    # 创建一个统一的图表
    fig = plt.figure(figsize=(16, rows * 6))
    
    for i, term in enumerate(model.terms):
        term_type = term.__class__.__name__
        
        # 根据项的类型创建不同的可视化
        if i == 0 and term_type == 'Intercept':
            # 跳过截距项
            continue
        
        # 创建子图
        ax = fig.add_subplot(rows, cols, i + 1)
        
        if term_type == 'SplineTerm':
            # 为平滑项创建一维偏依赖图
            feature_idx = term.feature
            feature_name = feature_names[feature_idx]
            
            # 获取XX网格以及预测和置信区间
            XX = model.generate_X_grid(term=i)
            pdep, confi = model.partial_dependence(term=i, X=XX, width=0.95)
            
            # 绘制曲线和置信区间
            ax.plot(XX[:, feature_idx], pdep)
            ax.fill_between(XX[:, feature_idx], confi[:, 0], confi[:, 1], alpha=0.3)
            
            # 设置标题和轴标签
            ax.set_title(f'{feature_name}的非线性影响', fontsize=14)
            ax.set_xlabel(feature_name, fontsize=12)
            ax.set_ylabel('Y染色体浓度的偏效应', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
        elif term_type == 'LinearTerm':
            # 为线性项创建一维偏依赖图
            feature_idx = term.feature
            feature_name = feature_names[feature_idx]
            
            # 获取XX网格以及预测
            XX = model.generate_X_grid(term=i)
            pdep, confi = model.partial_dependence(term=i, X=XX, width=0.95)
            
            # 绘制直线和置信区间
            ax.plot(XX[:, feature_idx], pdep)
            ax.fill_between(XX[:, feature_idx], confi[:, 0], confi[:, 1], alpha=0.3)
            
            # 设置标题和轴标签
            ax.set_title(f'{feature_name}的线性影响', fontsize=14)
            ax.set_xlabel(feature_name, fontsize=12)
            ax.set_ylabel('Y染色体浓度的偏效应', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
        elif term_type == 'TensorTerm':
            # 为交互项创建二维热力图/等高线图
            try:
                # 从 tensor term 的 info 中提取特征索引
                terms_info = term.info['terms']
                feat_idx1 = int(terms_info[0]['feature'])
                feat_idx2 = int(terms_info[1]['feature'])
            except (KeyError, IndexError, ValueError):
                # 如果无法获取特征索引，使用默认值
                feat_idx1, feat_idx2 = 0, 1
            
            feat_name1 = feature_names[feat_idx1] if feat_idx1 < len(feature_names) else f'feature_{feat_idx1}'
            feat_name2 = feature_names[feat_idx2] if feat_idx2 < len(feature_names) else f'feature_{feat_idx2}'
            
            # 创建特征1和特征2的网格
            XX = model.generate_X_grid(term=i, n=50)
            
            # 计算偏依赖
            Z = model.partial_dependence(term=i, X=XX)
            
            # 准备网格数据进行可视化
            x = XX[:, feat_idx1]
            y = XX[:, feat_idx2]
            
            # 重塑数据为网格形式
            n_unique_x = len(np.unique(x))
            n_unique_y = len(np.unique(y))
            X_mesh = x.reshape(n_unique_x, n_unique_y)
            Y_mesh = y.reshape(n_unique_x, n_unique_y)
            Z_mesh = Z.reshape(n_unique_x, n_unique_y)
            
            # 创建自定义蓝-白-红色映射
            colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # 蓝白红
            cmap_name = 'blue_white_red'
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
            
            # 绘制热力图
            c = ax.pcolormesh(X_mesh, Y_mesh, Z_mesh, cmap=cm, shading='auto')
            
            # 添加等高线
            contour = ax.contour(X_mesh, Y_mesh, Z_mesh, colors='k', alpha=0.5)
            ax.clabel(contour, inline=True, fontsize=8)
            
            # 添加颜色条
            plt.colorbar(c, ax=ax, label='Y染色体浓度的偏效应')
            
            # 设置标题和轴标签
            ax.set_title(f'{feat_name1}与{feat_name2}的交互效应', fontsize=14)
            ax.set_xlabel(feat_name1, fontsize=12)
            ax.set_ylabel(feat_name2, fontsize=12)
    
    # 调整子图布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f'{output_dir}/partial_dependence_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存偏依赖图到 {output_dir}/partial_dependence_plots.png")

def perform_model_diagnostics(model, X, y, feature_names, output_dir):
    """进行模型诊断：残差分析和QQ图"""
    
    # 获取预测值和残差
    if len(feature_names) != X.shape[1]:
        # 如果最佳模型只使用了部分特征，需要提取对应列
        indices = [list(X.columns).index(feat) for feat in feature_names]
        X_subset = X.iloc[:, indices]
        y_pred = model.predict(X_subset)
    else:
        y_pred = model.predict(X)
    
    residuals = y - y_pred
    
    # 创建包含两个子图的画布：残差vs拟合值图和QQ图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 残差vs拟合值图
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='-')
    
    # 添加平滑线来显示趋势
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(residuals, y_pred, frac=0.3)
        axes[0].plot(smooth[:, 0], smooth[:, 1], color='red', lw=2)
    except:
        pass  # 如果lowess导入失败，跳过平滑线
    
    axes[0].set_title('残差 vs. 拟合值图', fontsize=14)
    axes[0].set_xlabel('拟合值', fontsize=12)
    axes[0].set_ylabel('残差', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # 2. QQ图
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('残差正态Q-Q图', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存诊断图
    plt.savefig(f'{output_dir}/model_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存模型诊断图到 {output_dir}/model_diagnostics.png")
    
    # 计算并保存一些基本的诊断统计量
    diag_stats = pd.DataFrame({
        '指标': ['均方误差(MSE)', '平均绝对误差(MAE)', '残差均值', '残差标准差', '残差中位数'],
        '值': [
            np.mean(residuals**2),
            np.mean(np.abs(residuals)),
            np.mean(residuals),
            np.std(residuals),
            np.median(residuals)
        ]
    })
    
    diag_stats.to_csv(f'{output_dir}/diagnostic_statistics.csv', index=False, encoding='utf-8-sig')
    print(f"已保存诊断统计量到 {output_dir}/diagnostic_statistics.csv")

if __name__ == "__main__":
    gam_enhanced_modeling()
