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
import argparse
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from io import StringIO
import contextlib
import warnings
try:
    from statsmodels.stats.diagnostic import het_breuschpagan
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# 确保可以导入项目根目录的模块
sys.path.append('/Users/Mac/Downloads/mm')
from set_chinese_font import set_chinese_font

# 设置matplotlib以正确显示中文
set_chinese_font()

def gam_enhanced_modeling(n_splines_s=20, n_splines_te=10, k=5, repeats=3, save_fold_predictions=True):
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
    # 生成样本ID（优先使用‘孕妇代码’，否则使用行号），并与mask对齐
    if '孕妇代码' in data.columns:
        sample_ids = data['孕妇代码'].astype(str)
    else:
        sample_ids = pd.Series(np.arange(len(data)), name='row_index').astype(str)
    sample_ids = sample_ids[mask]
    
    print(f"清理后的建模数据样本数: {len(y)}")

    # 3. 构建和比较多个GAM模型（系统化候选 + λ调参）
    print("\n开始构建和比较多个GAM模型...")
    lam_grid = np.logspace(-3, 3, 7)
    specs = build_model_specs(feature_names, n_splines_s=n_splines_s, n_splines_te=n_splines_te)
    print(f"候选模型数量: {len(specs)}")
    models = {}
    model_features = {}

    for i, (name, spec) in enumerate(specs.items(), start=1):
        feats = spec['features']
        X_used = X[feats]
        print(f"  - 正在训练模型{i}/{len(specs)}: {name} ...")
        builder = spec['builder']
        gam = builder()
        try:
            gam = gam.gridsearch(X_used, y, lam=lam_grid, progress=False)
        except Exception as e:
            warnings.warn(f"模型{name} 使用gridsearch失败，改用直接拟合: {e}")
            try:
                gam.fit(X_used, y)
            except Exception as ee:
                warnings.warn(f"模型{name} 拟合失败：{ee}")
                continue
        models[name] = gam
        model_features[name] = feats

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
    
    # 根据最优模型选择对应的特征（来自specs映射）
    best_model_features = model_features.get(best_model_name, feature_names)
    
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

    # 9. 重复K折交叉验证评估所有候选模型（MAE/RMSE/MedAE/R²）并带λ调参
    print("\n开始进行重复5折交叉验证评估所有候选模型 (MAE/RMSE/MedAE/R²)...")
    cv_df = cross_validate_all_models(
        X,
        y,
        specs,
        k=k,
        repeats=repeats,
        random_state=42,
        lam_grid=lam_grid,
        save_fold_predictions=save_fold_predictions,
        output_dir=output_dir,
        sample_ids=sample_ids,
    )
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
    """进行模型诊断：残差分析、正态性与异方差检验"""
    
    # 获取预测值和残差
    if len(feature_names) != X.shape[1]:
        # 如果最佳模型只使用了部分特征，需要提取对应列
        indices = [list(X.columns).index(feat) for feat in feature_names]
        X_for_use = X.iloc[:, indices]
    else:
        X_for_use = X
    
    y_pred = model.predict(X_for_use)
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
    
    # 计算并保存诊断统计量 + 正态性与异方差检验
    try:
        shapiro_w, shapiro_p = stats.shapiro(residuals)
    except Exception:
        shapiro_w, shapiro_p = np.nan, np.nan
    try:
        jb_stat, jb_p = stats.jarque_bera(residuals)
    except Exception:
        jb_stat, jb_p = np.nan, np.nan
    bp_lm, bp_p = np.nan, np.nan
    if HAS_STATSMODELS:
        try:
            exog = np.column_stack([np.ones(len(X_for_use)), X_for_use.values])
            bp_lm, bp_lm_pvalue, _, _ = het_breuschpagan(residuals, exog)
            bp_p = bp_lm_pvalue
        except Exception:
            bp_lm, bp_p = np.nan, np.nan
    diag_stats = pd.DataFrame({
        '指标': [
            '均方误差(MSE)',
            '平均绝对误差(MAE)',
            '残差均值',
            '残差标准差',
            '残差中位数',
            'Shapiro-W (正态性)',
            'Shapiro-p',
            'Jarque-Bera',
            'Jarque-Bera p',
            'BP LM (异方差)',
            'BP p'
        ],
        '值': [
            np.mean(residuals**2),
            np.mean(np.abs(residuals)),
            np.mean(residuals),
            np.std(residuals),
            np.median(residuals),
            shapiro_w,
            shapiro_p,
            jb_stat,
            jb_p,
            bp_lm,
            bp_p
        ]
    })
    
    diag_stats.to_csv(f'{output_dir}/diagnostic_statistics.csv', index=False, encoding='utf-8-sig')
    print(f"已保存诊断统计量到 {output_dir}/diagnostic_statistics.csv")

def cross_validate_all_models(X, y, specs, k=5, repeats=3, random_state=42, lam_grid=None, save_fold_predictions=False, output_dir=None, sample_ids=None):
    """对候选模型进行重复K折交叉验证，返回每个模型的平均指标（含调参）
    可选：保存每折的预测与残差到 output_dir/cv_predictions.csv
    """
    n = len(y)
    rng_global = np.random.RandomState(random_state)

    # 为每个模型准备容器
    results = {name: {'MAE': [], 'MedAE': [], 'RMSE': [], 'R2': []} for name in specs}
    pred_rows = [] if save_fold_predictions else None

    for rep in range(repeats):
        idx = np.arange(n)
        rng = np.random.RandomState(rng_global.randint(0, 10**6))
        rng.shuffle(idx)

        # 计算每折大小
        fold_sizes = np.full(k, n // k, dtype=int)
        fold_sizes[: n % k] += 1
        current = 0

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
                preds = None
                try:
                    if lam_grid is not None:
                        try:
                            gam = gam.gridsearch(X_train, y_train, lam=lam_grid, progress=False)
                        except Exception:
                            # 尝试直接拟合
                            gam.fit(X_train, y_train)
                    else:
                        gam.fit(X_train, y_train)
                    preds = gam.predict(X_test)
                    mae = float(np.mean(np.abs(y_test - preds)))
                    medae = float(np.median(np.abs(y_test - preds)))
                    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
                    ss_res = float(np.sum((y_test - preds) ** 2))
                    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
                    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                except Exception:
                    # 失败时用 NaN 填充预测，保证后续保存流程稳定
                    preds = np.full(shape=len(y_test), fill_value=np.nan, dtype=float)
                    mae, medae, rmse, r2 = np.nan, np.nan, np.nan, np.nan
                results[name]['MAE'].append(mae)
                results[name]['MedAE'].append(medae)
                results[name]['RMSE'].append(rmse)
                results[name]['R2'].append(r2)
                if save_fold_predictions:
                    try:
                        ids = sample_ids.iloc[test_idx].values if sample_ids is not None else test_idx
                    except Exception:
                        ids = test_idx
                    pred_rows.append(pd.DataFrame({
                        'repeat': rep + 1,
                        'fold': fold + 1,
                        '模型': name,
                        'sample_id': ids,
                        'y_true': y_test.values,
                        'y_pred': preds,
                        'residual': y_test.values - preds
                    }))

    # 汇总
    rows = []
    for name, mets in results.items():
        row = {
            '模型': name,
            'MAE_mean': np.nanmean(mets['MAE']),
            'MAE_std': np.nanstd(mets['MAE']),
            'MedAE_mean': np.nanmean(mets['MedAE']),
            'MedAE_std': np.nanstd(mets['MedAE']),
            'RMSE_mean': np.nanmean(mets['RMSE']),
            'RMSE_std': np.nanstd(mets['RMSE']),
            'R2_mean': np.nanmean(mets['R2']),
            'R2_std': np.nanstd(mets['R2'])
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(by=['RMSE_mean', 'MAE_mean', 'R2_mean'], ascending=[True, True, False])
    for col in ['MAE_mean', 'MAE_std', 'MedAE_mean', 'MedAE_std', 'RMSE_mean', 'RMSE_std', 'R2_mean', 'R2_std']:
        df[col] = df[col].map(lambda v: f"{v:.6f}" if pd.notnull(v) else "")
    # 保存每折预测
    if save_fold_predictions and output_dir is not None and pred_rows:
        pred_df = pd.concat(pred_rows, ignore_index=True)
        pred_df.to_csv(f'{output_dir}/cv_predictions.csv', index=False, encoding='utf-8-sig')
    return df

def build_model_specs(feature_names, n_splines_s=20, n_splines_te=10):
    """构建系统化的候选模型规格：不强制引入l(3)/l(4)，按需比较；交互项包含单独与全对交互。
    n_splines_s: 单变量平滑项基函数数量
    n_splines_te: 交互张量项基函数数量（每维）
    """
    # 假设前3个是连续变量：孕周、BMI、年龄
    core3 = feature_names[:3]
    has_preg = len(feature_names) > 3
    has_birth = len(feature_names) > 4

    specs = {
        'M1_baseline': {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s)
            ),
            'features': core3
        },
        'M2_interact_WB': {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s) + te(0, 1, n_splines=n_splines_te)
            ),
            'features': core3
        },
        'M3_interact_WA': {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s) + te(0, 2, n_splines=n_splines_te)
            ),
            'features': core3
        },
        'M4_interact_BA': {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s) + te(1, 2, n_splines=n_splines_te)
            ),
            'features': core3
        },
        'M5_all_pairwise_interactions': {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s)
                + te(0, 1, n_splines=n_splines_te) + te(0, 2, n_splines=n_splines_te) + te(1, 2, n_splines=n_splines_te)
            ),
            'features': core3
        }
    }

    if has_preg:
        specs['M6_add_pregnancy_linear'] = {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s) + l(3)
            ),
            'features': core3 + [feature_names[3]]
        }
    if has_birth:
        specs['M7_add_birth_linear'] = {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s) + l(3)
            ),
            'features': core3 + [feature_names[4]]
        }
    if has_preg and has_birth:
        specs['M8_add_preg_birth_linear'] = {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s) + l(3) + l(4)
            ),
            'features': core3 + [feature_names[3], feature_names[4]]
        }
        specs['M9_full_main_linear_counts_all_interactions'] = {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s) + l(3) + l(4)
                + te(0, 1, n_splines=n_splines_te) + te(0, 2, n_splines=n_splines_te) + te(1, 2, n_splines=n_splines_te)
            ),
            'features': core3 + [feature_names[3], feature_names[4]]
        }
    return specs

 # 已移除 GLMGam 路线，保留单一 pygam LinearGAM 方案


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="已迁移：建议使用 main.py 运行。本入口为兼容包装。")
    parser.add_argument("--n_splines_s", type=int, default=20, help="单变量平滑项基函数数量")
    parser.add_argument("--n_splines_te", type=int, default=10, help="交互张量项每维基函数数量")
    parser.add_argument("--k", type=int, default=5, help="K折交叉验证的折数")
    parser.add_argument("--repeats", type=int, default=3, help="重复K折次数")
    parser.add_argument("--no_save_fold_predictions", action="store_true", help="不保存每折预测与残差")
    parser.add_argument("--processed_data", type=str, default="processed_data.csv")
    parser.add_argument("--output_dir", type=str, default="gam_enhanced_results")
    args = parser.parse_args()

    try:
        from main import run as _run
        _run(
            processed_data_path=args.processed_data,
            output_dir=args.output_dir,
            n_splines_s=args.n_splines_s,
            n_splines_te=args.n_splines_te,
            k=args.k,
            repeats=args.repeats,
            save_fold_predictions=not args.no_save_fold_predictions,
        )
    except Exception:
        # 兜底：仍可调用旧函数（不推荐）
        gam_enhanced_modeling(
            n_splines_s=args.n_splines_s,
            n_splines_te=args.n_splines_te,
            k=args.k,
            repeats=args.repeats,
            save_fold_predictions=not args.no_save_fold_predictions,
        )
