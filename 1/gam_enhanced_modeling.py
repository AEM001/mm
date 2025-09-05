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
from statsmodels.gam.api import GLMGam, BSplines
from scipy import stats
from io import StringIO
import contextlib

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
    
    # 清理数据中的NaN和Inf值
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y = y[mask]
    X = X[mask]
    
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

    # 模型3: 孕周×BMI交互 (在M2基础上添加交互项)
    print("  - 正在训练模型3 (M2 + 孕周×BMI交互)...")
    models['M3_interact_WB'] = LinearGAM(s(0) + s(1) + s(2) + l(3) + l(4) + te(0, 1)).fit(X, y)
    
    # 模型4: 孕周×年龄交互 (在M2基础上添加交互项)
    print("  - 正在训练模型4 (M2 + 孕周×年龄交互)...")
    models['M4_interact_WA'] = LinearGAM(s(0) + s(1) + s(2) + l(3) + l(4) + te(0, 2)).fit(X, y)

    # 模型5: BMI×年龄交互 (在M2基础上添加交互项)
    print("  - 正在训练模型5 (M2 + BMI×年龄交互)...")
    models['M5_interact_BA'] = LinearGAM(s(0) + s(1) + s(2) + l(3) + l(4) + te(1, 2)).fit(X, y)
    
    # 模型6: BMI+孕周主效应+交互项 (专门测试BMI、孕周及其交互)
    print("  - 正在训练模型6 (BMI + 孕周 + BMI×孕周交互)...")
    models['M6_WB_focused'] = LinearGAM(s(0) + s(1) + te(0, 1)).fit(X[['孕周_标准化', 'BMI_标准化']], y)

    # 4. 比较模型性能并选择最优模型
    print("\n模型性能比较:")
    results = []
    for name, model in models.items():
        # 计算显著性检验
        p_values = model.statistics_['p_values']
        # 排除截距项
        pvals_non_intercept = [p for t, p in zip(model.terms, p_values) if not t.isintercept]
        significant_terms = sum(p < 0.05 for p in pvals_non_intercept)
        
        results.append({
            '模型': name,
            '伪R²': f"{model.statistics_['pseudo_r2']['explained_deviance']:.4f}",
            'AICc': f"{model.statistics_['AICc']:.2f}",
            '显著项数': f"{significant_terms}/{len(pvals_non_intercept)}"
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # 根据AICc选择最优模型 (值越小越好)
    best_model_name = min(models, key=lambda k: models[k].statistics_['AICc'])
    best_model = models[best_model_name]
    print(f"\n最优模型 (基于AICc): {best_model_name}")

    # 5. 分析并可视化最优模型
    print(f"\n最优模型 ({best_model_name}) 摘要:")
    # pygam.summary() 直接打印到标准输出，这里捕获文本
    _buf = StringIO()
    with contextlib.redirect_stdout(_buf):
        best_model.summary()
    summary_str = _buf.getvalue()
    print(summary_str)
    
    # 根据最优模型选择对应的特征
    if best_model_name == 'M1_baseline':
        best_model_features = feature_names[:3]  # 只有前三个特征：孕周、BMI、年龄
    elif best_model_name == 'M6_WB_focused':
        best_model_features = ['孕周_标准化', 'BMI_标准化']  # M6模型只用孕周和BMI
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

    # 8. 整体模型的近似 F 检验（基于EDoF 和 方差分解，供参考）
    if len(best_model_features) != X.shape[1]:
        _idx = [list(X.columns).index(feat) for feat in best_model_features]
        X_used = X.iloc[:, _idx]
    else:
        X_used = X
    y_pred_best = best_model.predict(X_used)
    n = len(y)
    SSE = float(np.sum((y - y_pred_best) ** 2))
    SSR = float(np.sum((y_pred_best - y.mean()) ** 2))
    edof = float(best_model.statistics_['edof'])
    df1 = max(1, int(np.ceil(edof)))
    df2 = max(1, int(n - df1))
    F_stat = (SSR / df1) / (SSE / df2)
    F_p_value = float(stats.f.sf(F_stat, df1, df2))
    print(f"\n整体模型显著性近似F检验: F({df1}, {df2}) = {F_stat:.3f}, p = {F_p_value:.4g}")
    
    # 保存基本结果到文本文件
    with open(f'{output_dir}/best_model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"最优GAM模型分析结果: {best_model_name}\n")
        f.write("="*70 + "\n")
        f.write("模型比较:\n")
        f.write(results_df.to_string(index=False) + "\n\n")
        f.write("="*70 + "\n")
        f.write(f"最优模型 ({best_model_name}) 摘要:\n")
        f.write(summary_str + "\n")
        f.write("整体模型显著性近似F检验:\n")
        f.write(f"F({df1}, {df2}) = {F_stat:.3f}, p = {F_p_value:.4g}\n\n")
        f.write("注: GAM 平滑项的 p 值与上述 F 检验均为近似量，且 pyGAM 在估计平滑参数时 p 值可能偏小，请谨慎解读。\n")
        
    print(f"\n分析完成。所有结果已保存至 '{output_dir}' 目录。")

    # 9. K折交叉验证评估所有候选模型（MAE/RMSE/R²）
    print("\n开始进行5折交叉验证评估所有候选模型 (MAE/RMSE/R²)...")
    cv_df = cross_validate_all_models(X, y, feature_names, k=5, random_state=42)
    print("\n交叉验证汇总结果:")
    print(cv_df.to_string(index=False))
    cv_df.to_csv(f'{output_dir}/cv_results.csv', index=False, encoding='utf-8-sig')
    print(f"已保存交叉验证结果至 {output_dir}/cv_results.csv")

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
        
        # 获取Lambda (平滑惩罚系数)：对数组保留完整显示
        lam_display = ""
        try:
            if hasattr(model, 'lam') and i < len(model.lam):
                lam_val = model.lam[i]
                if isinstance(lam_val, (list, tuple, np.ndarray)):
                    lam_display = np.array2string(np.array(lam_val), precision=1, separator=' ')
                elif isinstance(lam_val, (int, float, np.floating)):
                    lam_display = f"{lam_val:.6f}"
                else:
                    lam_display = str(lam_val)
        except Exception:
            lam_display = ""
        
        # 添加行数据
        rows.append({
            'Term (模型项)': term_name,
            'Term_Type (项类型)': term_type_cn,
            'EDoF (有效自由度)': f"{edof:.3f}" if not np.isnan(edof) else "",
            'Lambda (平滑惩罚系数)': lam_display,
            'P-value (P值)': f"{p_value:.4f}" if not np.isnan(p_value) else "",
            'Significance (显著性水平)': significance,
            'Interpretation (业务解读)': "" # 留空，用于后续手动填写
        })
    
    # 创建DataFrame
    return pd.DataFrame(rows)

def create_partial_dependence_plots(model, feature_names, output_dir):
    """为模型中的每个项创建偏依赖图"""
    
    # 收集需要绘制的项（排除截距），并计算网格
    plot_indices = [i for i, t in enumerate(model.terms) if not getattr(t, 'isintercept', False)]
    num_plots = len(plot_indices)
    rows = int(np.ceil(num_plots / 2))
    cols = 2 if num_plots > 1 else 1
    fig = plt.figure(figsize=(16, rows * 6))

    for plot_pos, i in enumerate(plot_indices, start=1):
        term = model.terms[i]
        term_type = term.__class__.__name__
        
        # 创建子图（位置由实际绘制的项数决定，避免空白子图）
        ax = fig.add_subplot(rows, cols, plot_pos)
        
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

def cross_validate_all_models(X, y, feature_names, k=5, random_state=42):
    """对当前脚本中的6个候选模型进行K折交叉验证，返回每个模型的平均指标"""
    # 定义模型规格（构建器 + 使用的特征列表）
    specs = {
        'M1_baseline': {
            'builder': lambda: LinearGAM(s(0) + s(1) + s(2)),
            'features': feature_names[:3]
        },
        'M2_add_C_D': {
            'builder': lambda: LinearGAM(s(0) + s(1) + s(2) + l(3) + l(4)),
            'features': feature_names
        },
        'M3_interact_WB': {
            'builder': lambda: LinearGAM(s(0) + s(1) + s(2) + l(3) + l(4) + te(0, 1)),
            'features': feature_names
        },
        'M4_interact_WA': {
            'builder': lambda: LinearGAM(s(0) + s(1) + s(2) + l(3) + l(4) + te(0, 2)),
            'features': feature_names
        },
        'M5_interact_BA': {
            'builder': lambda: LinearGAM(s(0) + s(1) + s(2) + l(3) + l(4) + te(1, 2)),
            'features': feature_names
        },
        'M6_WB_focused': {
            'builder': lambda: LinearGAM(s(0) + s(1) + te(0, 1)),
            'features': ['孕周_标准化', 'BMI_标准化']
        }
    }

    n = len(y)
    idx = np.arange(n)
    rng = np.random.RandomState(random_state)
    rng.shuffle(idx)

    # 计算每折大小
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    current = 0

    # 为每个模型准备容器
    results = {name: {'MAE': [], 'RMSE': [], 'R2': []} for name in specs}

    for fold in range(k):
        start, stop = current, current + fold_sizes[fold]
        test_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]])
        current = stop

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        for name, spec in specs.items():
            feats = spec['features']
            X_train = X[feats].iloc[train_idx]
            X_test = X[feats].iloc[test_idx]
            gam = spec['builder']()
            try:
                gam.fit(X_train, y_train)
                preds = gam.predict(X_test)
                mae = float(np.mean(np.abs(y_test - preds)))
                rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
                # R² on test fold
                ss_res = float(np.sum((y_test - preds) ** 2))
                ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
                r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
            except Exception as e:
                mae, rmse, r2 = np.nan, np.nan, np.nan
            results[name]['MAE'].append(mae)
            results[name]['RMSE'].append(rmse)
            results[name]['R2'].append(r2)

    # 汇总
    rows = []
    for name, mets in results.items():
        row = {
            '模型': name,
            'MAE_mean': np.nanmean(mets['MAE']),
            'MAE_std': np.nanstd(mets['MAE']),
            'RMSE_mean': np.nanmean(mets['RMSE']),
            'RMSE_std': np.nanstd(mets['RMSE']),
            'R2_mean': np.nanmean(mets['R2']),
            'R2_std': np.nanstd(mets['R2'])
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    # 排序：优先RMSE，其次MAE
    df = df.sort_values(by=['RMSE_mean', 'MAE_mean', 'R2_mean'], ascending=[True, True, False])
    # 保留小数
    for col in ['MAE_mean', 'MAE_std', 'RMSE_mean', 'RMSE_std', 'R2_mean', 'R2_std']:
        df[col] = df[col].map(lambda v: f"{v:.6f}" if pd.notnull(v) else "")
    return df

# ===============
# 路线B：Fractional-Logit（Binomial + logit）GAM（statsmodels GLMGam + BSplines）
# ===============

def gam_fractional_logit_modeling():
    """
    使用 statsmodels 的 GLMGam（Binomial 家族 + logit 链接）对比例型响应进行加性建模。
    - 使用 BSplines 作为平滑器
    - 在比例刻度上评估（MAE/RMSE/R²）
    - 生成偏依赖图与诊断图
    - 使用 K 折交叉验证评估泛化性能
    输出目录：gam_enhanced_results_glm
    """
    output_dir = 'gam_enhanced_results_glm'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1) 读取数据
    try:
        df = pd.read_csv('processed_data.csv')
    except FileNotFoundError:
        return

    # 目标：比例，裁剪到 (0,1)
    if 'Y染色体浓度' not in df.columns:
        return
    eps = 1e-6
    y = df['Y染色体浓度'].clip(eps, 1 - eps)

    # 核心特征（仅使用5个关键临床变量）
    candidate_features = [
        '孕周_标准化', 'BMI_标准化', '年龄_标准化',
        '怀孕次数_标准化', '生产次数_标准化'
    ]
    features = [c for c in candidate_features if c in df.columns]
    if len(features) == 0:
        return

    X = df[features].copy()

    # 清理 NaN/Inf
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y = y[mask]
    X = X[mask]

    # 2) 配置 BSplines（统一自由度设置，简化模型复杂度）
    df_value = 6  # 统一设置为6个自由度，平衡拟合能力与泛化性能
    df_list = [df_value] * X.shape[1]
    bs = BSplines(X, df=df_list, degree=[3] * X.shape[1])

    # 3) 拟合 Fractional-Logit GAM
    model = GLMGam(y, smoother=bs, family=sm.families.Binomial())
    res = model.fit()

    # 4) 训练内评估（比例刻度）
    y_hat = res.predict(exog=None, exog_smooth=X, transform=True)
    mae = float(np.mean(np.abs(y - y_hat)))
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # 伪R²（解释偏差）
    try:
        # 若 GLMGamResults 未提供 null_deviance，则用截距-only 的 GLM 估计
        null_mod = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial()).fit()
        dev_null = float(null_mod.deviance)
        dev_model = float(res.deviance)
        pseudo_r2 = float(1 - dev_model / dev_null) if dev_null > 0 else np.nan
    except Exception:
        pseudo_r2 = np.nan

    # 5) 输出摘要
    try:
        summary_str = res.summary().as_text()
    except Exception:
        # 某些环境下 as_text 可能不可用，退回 str
        summary_str = str(res.summary())

    with open(f'{output_dir}/best_model_summary_glm.txt', 'w', encoding='utf-8') as f:
        f.write("Fractional-Logit GAM（Binomial + logit）模型摘要\n")
        f.write("=" * 70 + "\n")
        f.write(summary_str + "\n\n")
        f.write("训练集评估（比例刻度）:\n")
        f.write(f"MAE = {mae:.6f}\nRMSE = {rmse:.6f}\nR² = {r2:.6f}\n")
        f.write(f"解释偏差（Pseudo R² by deviance）= {pseudo_r2:.6f}\n")

    # 6) 偏依赖图（逐变量，其他变量固定在中位数）
    create_partial_dependence_plots_glm(res, X, output_dir)

    # 7) 诊断图与诊断统计
    perform_model_diagnostics_glm(y, y_hat, output_dir)

    # 8) 交叉验证（比例刻度上评估）
    cv_df = cross_validate_glmgam(X, y, df_list=df_list, degree=[3] * X.shape[1], k=5, random_state=42)
    cv_df.to_csv(f'{output_dir}/cv_results_glm.csv', index=False, encoding='utf-8-sig')
    
    # 9) 输出详细模型信息
    export_model_details(res, X, y, y_hat, mae, rmse, r2, pseudo_r2, cv_df, output_dir)


def create_partial_dependence_plots_glm(res, X, output_dir):
    """
    为 GLMGam 模型创建偏依赖图：
    - 对每个特征 i，构造该特征的一维网格，其余特征固定为中位数
    - 调用 res.predict 计算预测均值
    - 注意：此处不绘制置信区间（GLMGam 的预测方差获取较复杂）
    """
    feature_names = X.columns.tolist()
    n_features = len(feature_names)
    rows = int(np.ceil(n_features / 2))
    cols = 2 if n_features > 1 else 1
    fig = plt.figure(figsize=(16, rows * 6))

    # 基准点：中位数
    med = X.median(axis=0)

    for i, name in enumerate(feature_names, start=1):
        ax = fig.add_subplot(rows, cols, i)
        grid = np.linspace(X[name].min(), X[name].max(), 120)
        X_grid = pd.DataFrame(np.tile(med.values, (len(grid), 1)), columns=feature_names)
        X_grid[name] = grid
        # 预测
        y_grid = res.predict(exog=None, exog_smooth=X_grid, transform=True)
        ax.plot(grid, y_grid, color='C0')
        ax.set_title(f'{name} 的偏效应（比例刻度）', fontsize=14)
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel('预测的 Y染色体浓度', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/partial_dependence_plots_glm.png', dpi=300, bbox_inches='tight')
    plt.close()


def perform_model_diagnostics_glm(y_true, y_pred, output_dir):
    """GLMGam 的基本诊断：残差 vs 拟合值 + QQ 图 + 诊断统计输出"""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # 残差 vs 拟合值
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='-')
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(residuals, y_pred, frac=0.3)
        axes[0].plot(smooth[:, 0], smooth[:, 1], color='red', lw=2)
    except Exception:
        pass
    axes[0].set_title('残差 vs. 拟合值图（GLMGam）', fontsize=14)
    axes[0].set_xlabel('拟合值（比例）', fontsize=12)
    axes[0].set_ylabel('残差', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    # QQ 图
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('残差正态Q-Q图（GLMGam）', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_diagnostics_glm.png', dpi=300, bbox_inches='tight')
    plt.close()

    diag_stats = pd.DataFrame({
        '指标': ['均方误差(MSE)', '平均绝对误差(MAE)', '残差均值', '残差标准差', '残差中位数'],
        '值': [
            float(np.mean(residuals ** 2)),
            float(np.mean(np.abs(residuals))),
            float(np.mean(residuals)),
            float(np.std(residuals)),
            float(np.median(residuals))
        ]
    })
    diag_stats.to_csv(f'{output_dir}/diagnostic_statistics_glm.csv', index=False, encoding='utf-8-sig')


def export_model_details(res, X, y, y_hat, mae, rmse, r2, pseudo_r2, cv_df, output_dir):
    """
    输出详细的模型信息到CSV文件，包括系数、统计量和性能指标
    """
    # 1) 模型基本信息
    model_info = {
        '模型类型': 'Fractional-Logit GAM',
        '分布家族': 'Binomial',
        '链接函数': 'Logit',
        '样本数': len(y),
        '特征数': X.shape[1],
        '模型自由度': res.df_model,
        '残差自由度': res.df_resid,
        '迭代次数': getattr(res, 'n_iter', 'N/A'),
        '收敛状态': '成功' if res.converged else '失败'
    }
    
    # 2) 性能指标
    performance_metrics = {
        '训练集_MAE': f'{mae:.6f}',
        '训练集_RMSE': f'{rmse:.6f}',
        '训练集_R²': f'{r2:.6f}',
        '解释偏差_伪R²': f'{pseudo_r2:.6f}',
        '对数似然': f'{res.llf:.4f}',
        'AIC': f'{res.aic:.4f}',
        'BIC': f'{res.bic:.4f}',
        '偏差': f'{res.deviance:.4f}',
        'Pearson_卡方': f'{res.pearson_chi2:.4f}'
    }
    
    # 3) 交叉验证结果
    cv_metrics = {
        'CV_MAE_均值': cv_df['MAE_mean'].iloc[0],
        'CV_MAE_标准差': cv_df['MAE_std'].iloc[0],
        'CV_RMSE_均值': cv_df['RMSE_mean'].iloc[0],
        'CV_RMSE_标准差': cv_df['RMSE_std'].iloc[0],
        'CV_R²_均值': cv_df['R2_mean'].iloc[0],
        'CV_R²_标准差': cv_df['R2_std'].iloc[0]
    }
    
    # 4) 特征系数与显著性
    feature_names = X.columns.tolist()
    coef_data = []
    
    # 获取系数和统计量
    for i, coef_name in enumerate(res.params.index):
        coef_value = res.params.iloc[i]
        std_err = res.bse.iloc[i] if hasattr(res, 'bse') else np.nan
        z_value = res.tvalues.iloc[i] if hasattr(res, 'tvalues') else np.nan
        p_value = res.pvalues.iloc[i] if hasattr(res, 'pvalues') else np.nan
        
        # 显著性标识
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = 'ns'
            
        coef_data.append({
            '系数名称': coef_name,
            '系数值': f'{coef_value:.6f}',
            '标准误差': f'{std_err:.6f}' if not np.isnan(std_err) else 'N/A',
            'Z统计量': f'{z_value:.3f}' if not np.isnan(z_value) else 'N/A',
            'P值': f'{p_value:.6f}' if not np.isnan(p_value) else 'N/A',
            '显著性': significance
        })
    
    # 5) 保存到文件
    # 模型信息
    model_info_df = pd.DataFrame([model_info])
    model_info_df.to_csv(f'{output_dir}/model_basic_info.csv', index=False, encoding='utf-8-sig')
    
    # 性能指标
    performance_df = pd.DataFrame([{**performance_metrics, **cv_metrics}])
    performance_df.to_csv(f'{output_dir}/model_performance.csv', index=False, encoding='utf-8-sig')
    
    # 系数表
    coef_df = pd.DataFrame(coef_data)
    coef_df.to_csv(f'{output_dir}/model_coefficients.csv', index=False, encoding='utf-8-sig')
    
    # 特征重要性汇总（按特征分组统计）
    feature_summary = []
    for feat in feature_names:
        # 找到该特征对应的所有系数项（样条基函数）
        feat_coeffs = [row for row in coef_data if feat in row['系数名称']]
        n_significant = sum(1 for row in feat_coeffs if row['显著性'] != 'ns')
        
        feature_summary.append({
            '特征名称': feat,
            '系数项数': len(feat_coeffs),
            '显著项数': n_significant,
            '显著比例': f'{n_significant/len(feat_coeffs):.2%}' if feat_coeffs else '0%'
        })
    
    feature_summary_df = pd.DataFrame(feature_summary)
    feature_summary_df.to_csv(f'{output_dir}/feature_importance_summary.csv', index=False, encoding='utf-8-sig')


def cross_validate_glmgam(X, y, df_list, degree, k=5, random_state=42):
    """
    对 GLMGam 进行 K 折交叉验证。
    - 每一折：用训练集构造 BSplines（避免信息泄露），拟合 GLMGam；在测试集上评估 MAE/RMSE/R²（比例刻度）。
    - 返回单模型的汇总 DataFrame（与现有风格一致）。
    """
    n = len(y)
    idx = np.arange(n)
    rng = np.random.RandomState(random_state)
    rng.shuffle(idx)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    current = 0

    maes, rmses, r2s = [], [], []

    for fold in range(k):
        start, stop = current, current + fold_sizes[fold]
        test_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]])
        current = stop

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # 为训练集构造样条并拟合
        bs_train = BSplines(X_train, df=df_list, degree=degree)
        model = GLMGam(y_train, smoother=bs_train, family=sm.families.Binomial())
        try:
            res = model.fit()
            # 将测试集数值裁剪到训练集范围内，避免样条外推报错
            train_min = X_train.min(axis=0)
            train_max = X_train.max(axis=0)
            X_test_clip = X_test.clip(lower=train_min, upper=train_max, axis=1)
            preds = res.predict(exog=None, exog_smooth=X_test_clip, transform=True)
            mae = float(np.mean(np.abs(y_test - preds)))
            rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
            ss_res = float(np.sum((y_test - preds) ** 2))
            ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        except Exception as e:
            mae, rmse, r2 = np.nan, np.nan, np.nan
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    df_result = pd.DataFrame([{
        '模型': 'GLMGam_Binomial_logit',
        'MAE_mean': np.nanmean(maes),
        'MAE_std': np.nanstd(maes),
        'RMSE_mean': np.nanmean(rmses),
        'RMSE_std': np.nanstd(rmses),
        'R2_mean': np.nanmean(r2s),
        'R2_std': np.nanstd(r2s)
    }])
    for col in ['MAE_mean', 'MAE_std', 'RMSE_mean', 'RMSE_std', 'R2_mean', 'R2_std']:
        df_result[col] = df_result[col].map(lambda v: f"{v:.6f}" if pd.notnull(v) else "")
    return df_result


if __name__ == "__main__":
    # 默认执行路线B（GLMGam），保留原pyGAM函数不动，避免大幅改动
    gam_fractional_logit_modeling()
