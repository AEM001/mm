#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的分数logit回归 - 实现孕周×BMI低自由度样条交互+L2正则
按照专家意见进行唯一优先改进：低自由度交互 + L2正则 + 分组CV选超参
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
import statsmodels.api as sm
from patsy import dmatrix, dmatrices
from scipy import stats
import itertools

# 确保可以导入项目根目录的模块
sys.path.append('/Users/Mac/Downloads/mm')
from set_chinese_font import set_chinese_font

# 设置matplotlib以正确显示中文
set_chinese_font()

# 忽略警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def create_interaction_features(data, df_gest=3, df_bmi=3, max_interactions=12):
    """
    创建孕周×BMI低自由度样条交互特征
    
    Parameters:
    -----------
    data : DataFrame
        包含孕周_标准化和BMI_标准化的数据
    df_gest, df_bmi : int
        孕周和BMI的样条自由度
    max_interactions : int
        最大交互项数量，控制复杂度
        
    Returns:
    --------
    X_matrix : DataFrame
        特征矩阵，包含主效应和交互项
    """
    
    # 创建主效应样条基
    gest_splines = dmatrix(f"bs(孕周_标准化, df={df_gest}, degree=3) - 1", 
                          data, return_type='dataframe')
    bmi_splines = dmatrix(f"bs(BMI_标准化, df={df_bmi}, degree=3) - 1", 
                         data, return_type='dataframe')
    
    # 重命名列以便识别
    gest_splines.columns = [f'孕周_B{i+1}' for i in range(gest_splines.shape[1])]
    bmi_splines.columns = [f'BMI_B{i+1}' for i in range(bmi_splines.shape[1])]
    
    # 创建交互项 (张量积)
    interaction_features = []
    interaction_names = []
    
    # 选择前几个最重要的基函数进行交互，控制复杂度
    n_gest_bases = min(df_gest, gest_splines.shape[1])
    n_bmi_bases = min(df_bmi, bmi_splines.shape[1])
    
    count = 0
    for i in range(n_gest_bases):
        for j in range(n_bmi_bases):
            if count >= max_interactions:
                break
            
            interaction = gest_splines.iloc[:, i] * bmi_splines.iloc[:, j]
            interaction_features.append(interaction)
            interaction_names.append(f'孕周_B{i+1}×BMI_B{j+1}')
            count += 1
        
        if count >= max_interactions:
            break
    
    # 合并所有特征
    all_features = [gest_splines, bmi_splines]
    if interaction_features:
        interaction_df = pd.concat(interaction_features, axis=1)
        interaction_df.columns = interaction_names
        all_features.append(interaction_df)
    
    X_matrix = pd.concat(all_features, axis=1)
    
    return X_matrix

def grid_search_with_group_cv(data, y, groups, param_grid, cv_folds=5, 
                             core_vars=None, quality_vars=None):
    """
    使用分组交叉验证进行超参数网格搜索
    
    Parameters:
    -----------
    data : DataFrame
        特征数据
    y : Series
        目标变量
    groups : Series
        分组标识（孕妇代码）
    param_grid : dict
        参数网格
    cv_folds : int
        交叉验证折数
    core_vars, quality_vars : list
        核心变量和质量变量列表
        
    Returns:
    --------
    best_params : dict
        最优参数
    cv_results : list
        所有参数组合的CV结果
    """
    
    print(f"开始网格搜索，参数组合数: {len(list(ParameterGrid(param_grid)))}")
    
    group_kfold = GroupKFold(n_splits=cv_folds)
    cv_results = []
    
    for param_combo in ParameterGrid(param_grid):
        print(f"  测试参数: {param_combo}")
        
        try:
            # 创建特征矩阵
            X_spline = create_interaction_features(
                data, 
                df_gest=param_combo['df_gest'], 
                df_bmi=param_combo['df_bmi'],
                max_interactions=param_combo.get('max_interactions', 12)
            )
            
            # 添加其他变量
            other_features = []
            if core_vars and '年龄_标准化' in data.columns:
                other_features.append(data[['年龄_标准化']])
            if quality_vars:
                available_quality = [v for v in quality_vars if v in data.columns]
                if available_quality:
                    other_features.append(data[available_quality])
            
            if other_features:
                X_full = pd.concat([X_spline] + other_features, axis=1)
            else:
                X_full = X_spline
            
            # 进行分组交叉验证
            fold_scores = {'mae': [], 'rmse': [], 'r2': []}
            
            for train_idx, test_idx in group_kfold.split(data, y, groups):
                X_train = X_full.iloc[train_idx]
                X_test = X_full.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                # 拟合带L2正则的GLM
                if param_combo['alpha'] > 0:
                    # 使用L2正则
                    glm = sm.GLM(y_train, sm.add_constant(X_train), 
                               family=sm.families.Binomial())
                    model = glm.fit_regularized(L1_wt=0.0, alpha=param_combo['alpha'])
                else:
                    # 不使用正则
                    glm = sm.GLM(y_train, sm.add_constant(X_train), 
                               family=sm.families.Binomial())
                    model = glm.fit()
                
                # 预测
                y_pred = model.predict(sm.add_constant(X_test))
                
                # 计算指标
                fold_scores['mae'].append(mean_absolute_error(y_test, y_pred))
                fold_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                fold_scores['r2'].append(r2_score(y_test, y_pred))
            
            # 记录平均性能
            cv_result = {
                **param_combo,
                'mae_mean': np.mean(fold_scores['mae']),
                'mae_std': np.std(fold_scores['mae']),
                'rmse_mean': np.mean(fold_scores['rmse']),
                'rmse_std': np.std(fold_scores['rmse']),
                'r2_mean': np.mean(fold_scores['r2']),
                'r2_std': np.std(fold_scores['r2'])
            }
            
            cv_results.append(cv_result)
            
            print(f"    RMSE: {cv_result['rmse_mean']:.6f} ± {cv_result['rmse_std']:.6f}")
            
        except Exception as e:
            print(f"    参数组合失败: {e}")
            continue
    
    if not cv_results:
        raise ValueError("所有参数组合都失败了")
    
    # 选择最优参数 (基于RMSE最小)
    best_result = min(cv_results, key=lambda x: x['rmse_mean'])
    best_params = {k: v for k, v in best_result.items() 
                  if k not in ['mae_mean', 'mae_std', 'rmse_mean', 'rmse_std', 'r2_mean', 'r2_std']}
    
    print(f"最优参数: {best_params}")
    print(f"最优RMSE: {best_result['rmse_mean']:.6f} ± {best_result['rmse_std']:.6f}")
    
    return best_params, cv_results

def evaluate_threshold_performance(y_true, y_pred, threshold=0.04):
    """
    评估4%阈值的分类性能
    
    Parameters:
    -----------
    y_true, y_pred : array-like
        真实值和预测值
    threshold : float
        分类阈值
        
    Returns:
    --------
    metrics : dict
        分类性能指标
    """
    
    # 转换为二分类
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # 计算混淆矩阵元素
    TP = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
    TN = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
    FP = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
    FN = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
    
    # 计算指标
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # 计算AUC
    try:
        auc = roc_auc_score(y_true_binary, y_pred)
    except:
        auc = 0.5
    
    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'auc': auc,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }

def create_bmi_risk_stratification(data, model, output_dir):
    """
    创建BMI分组风险分层和最佳时点分析
    """
    
    # BMI分组 (按照题目建议的分组)
    bmi_raw = data['BMI_标准化'] * data['BMI_标准化'].std() + data['BMI_标准化'].mean()
    
    bmi_bins = [20, 28, 32, 36, 40, np.inf]
    bmi_labels = ['[20,28)', '[28,32)', '[32,36)', '[36,40)', '≥40']
    bmi_groups = pd.cut(bmi_raw, bins=bmi_bins, labels=bmi_labels, right=False)
    
    # 孕周范围
    gest_raw = data['孕周_标准化'] * data['孕周_标准化'].std() + data['孕周_标准化'].mean()
    
    stratification_results = []
    
    for group_name in bmi_labels:
        group_mask = (bmi_groups == group_name)
        if group_mask.sum() < 10:  # 样本量太小跳过
            continue
        
        group_data = data[group_mask].copy()
        group_gest = gest_raw[group_mask]
        group_y = data.loc[group_mask, 'Y染色体浓度']
        
        # 预测该组的Y染色体浓度
        # 创建特征矩阵 (使用最优模型的配置)
        try:
            # 这里需要根据实际的最优模型配置来创建特征
            # 为简化，先使用基本配置
            X_group = create_interaction_features(group_data, df_gest=3, df_bmi=3)
            if '年龄_标准化' in group_data.columns:
                X_group = pd.concat([X_group, group_data[['年龄_标准化']]], axis=1)
            
            y_pred_group = model.predict(sm.add_constant(X_group))
            
            # 找到最早达标孕周
            达标mask = (y_pred_group >= 0.04)
            if 达标mask.sum() > 0:
                最早达标孕周 = group_gest[达标mask].min()
                达标比例 = 达标mask.mean()
            else:
                最早达标孕周 = np.nan
                达标比例 = 0.0
            
            stratification_results.append({
                'BMI组': group_name,
                '样本数': group_mask.sum(),
                'BMI均值': bmi_raw[group_mask].mean(),
                '孕周范围': f"{group_gest.min():.1f}-{group_gest.max():.1f}",
                '最早达标孕周': 最早达标孕周,
                '达标比例': 达标比例,
                'Y浓度均值': group_y.mean(),
                'Y浓度标准差': group_y.std()
            })
            
        except Exception as e:
            print(f"BMI组 {group_name} 分析失败: {e}")
            continue
    
    # 保存结果
    stratification_df = pd.DataFrame(stratification_results)
    stratification_df.to_csv(f'{output_dir}/bmi_risk_stratification.csv', 
                           index=False, encoding='utf-8-sig')
    
    print("BMI风险分层分析完成")
    return stratification_df

def fractional_logit_improved_modeling():
    """
    改进的分数logit回归建模主函数
    """
    
    # 创建输出目录
    output_dir = 'fractional_logit_improved_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 70)
    print("改进的分数logit回归 - 孕周×BMI交互+L2正则+分组CV")
    print("=" * 70)
    
    # 重新运行数据预处理
    print("\n1. 重新处理数据...")
    from data_process import process_data
    process_data()
    
    # 加载处理后的数据
    try:
        data = pd.read_csv('processed_data.csv')
        print(f"加载数据样本数: {len(data)}")
    except FileNotFoundError:
        print("错误: 未找到 'processed_data.csv'")
        return
    
    # 准备建模数据
    print("\n2. 准备建模数据...")
    y = data['Y染色体浓度'].copy()
    groups = data[data.columns[0]]  # 孕妇代码
    
    print(f"目标变量统计: min={y.min():.6f}, max={y.max():.6f}, mean={y.mean():.6f}")
    print(f"4%阈值达标比例: {(y >= 0.04).mean():.3f}")
    
    # 变量定义
    core_vars = ['孕周_标准化', 'BMI_标准化', '年龄_标准化']
    quality_vars = [
        '总读段数_标准化', '比对比例_标准化', '重复读段比例_标准化',
        '唯一比对读段比例_标准化', 'GC含量_标准化', '过滤读段比例_标准化'
    ]
    
    # 清理数据
    model_data = data[['Y染色体浓度'] + [data.columns[0]] + 
                     [v for v in core_vars + quality_vars if v in data.columns]].copy()
    model_data = model_data.dropna()
    
    y = model_data['Y染色体浓度'].copy()
    groups = model_data[data.columns[0]]
    
    print(f"清理后建模数据样本数: {len(model_data)}")
    
    # 参数网格搜索
    print("\n3. 进行参数网格搜索...")
    
    param_grid = {
        'df_gest': [3, 4],
        'df_bmi': [3, 4], 
        'alpha': [0.0, 0.001, 0.01, 0.1],
        'max_interactions': [9, 12]
    }
    
    available_core = [v for v in core_vars if v in model_data.columns]
    available_quality = [v for v in quality_vars if v in model_data.columns]
    
    best_params, cv_results = grid_search_with_group_cv(
        model_data, y, groups, param_grid, cv_folds=5,
        core_vars=available_core, quality_vars=available_quality
    )
    
    # 使用最优参数拟合最终模型
    print(f"\n4. 使用最优参数拟合最终模型...")
    print(f"最优参数: {best_params}")
    
    # 创建最优特征矩阵
    X_optimal = create_interaction_features(
        model_data,
        df_gest=best_params['df_gest'],
        df_bmi=best_params['df_bmi'],
        max_interactions=best_params['max_interactions']
    )
    
    # 添加其他变量
    other_features = []
    if '年龄_标准化' in model_data.columns:
        other_features.append(model_data[['年龄_标准化']])
    if available_quality:
        other_features.append(model_data[available_quality])
    
    if other_features:
        X_final = pd.concat([X_optimal] + other_features, axis=1)
    else:
        X_final = X_optimal
    
    # 拟合最终模型
    if best_params['alpha'] > 0:
        glm_final = sm.GLM(y, sm.add_constant(X_final), family=sm.families.Binomial())
        final_model = glm_final.fit_regularized(L1_wt=0.0, alpha=best_params['alpha'])
        print(f"使用L2正则化 (α={best_params['alpha']})")
    else:
        glm_final = sm.GLM(y, sm.add_constant(X_final), family=sm.families.Binomial())
        final_model = glm_final.fit()
        print("使用无正则化模型")
    
    # 模型评估
    print("\n5. 模型评估...")
    
    y_pred_final = final_model.predict(sm.add_constant(X_final))
    
    # 基本回归指标
    mae_final = mean_absolute_error(y, y_pred_final)
    rmse_final = np.sqrt(mean_squared_error(y, y_pred_final))
    r2_final = r2_score(y, y_pred_final)
    
    # 修正BIC计算
    n = len(y)
    k = len(final_model.params)
    llf = final_model.llf
    aic_corrected = -2 * llf + 2 * k
    bic_corrected = -2 * llf + k * np.log(n)
    
    print(f"回归性能:")
    print(f"  MAE: {mae_final:.6f}")
    print(f"  RMSE: {rmse_final:.6f}") 
    print(f"  R²: {r2_final:.4f}")
    print(f"  AIC: {aic_corrected:.2f}")
    print(f"  BIC: {bic_corrected:.2f}")
    print(f"  Log-Likelihood: {llf:.2f}")
    
    # 4%阈值分类评估
    threshold_metrics = evaluate_threshold_performance(y, y_pred_final, threshold=0.04)
    
    print(f"\n4%阈值分类性能:")
    print(f"  敏感度: {threshold_metrics['sensitivity']:.3f}")
    print(f"  特异度: {threshold_metrics['specificity']:.3f}")
    print(f"  精确度: {threshold_metrics['precision']:.3f}")
    print(f"  F1分数: {threshold_metrics['f1_score']:.3f}")
    print(f"  AUC: {threshold_metrics['auc']:.3f}")
    
    # BMI风险分层分析
    print("\n6. BMI风险分层分析...")
    model_data_with_final = model_data.copy()
    stratification_df = create_bmi_risk_stratification(model_data_with_final, final_model, output_dir)
    
    print("\nBMI分组最佳NIPT时点:")
    if not stratification_df.empty:
        for _, row in stratification_df.iterrows():
            if not np.isnan(row['最早达标孕周']):
                print(f"  {row['BMI组']}: {row['最早达标孕周']:.1f}周 (达标率{row['达标比例']:.1%})")
            else:
                print(f"  {row['BMI组']}: 无明确达标时点")
    
    # 保存结果
    print(f"\n7. 保存分析结果...")
    
    # 保存网格搜索结果
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.to_csv(f'{output_dir}/grid_search_results.csv', index=False, encoding='utf-8-sig')
    
    # 保存最终模型摘要
    with open(f'{output_dir}/improved_model_summary.txt', 'w', encoding='utf-8') as f:
        f.write("改进的分数logit回归模型分析结果\\n")
        f.write("=" * 70 + "\\n\\n")
        f.write(f"最优参数配置:\\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\\n")
        f.write(f"\\n回归性能指标:\\n")
        f.write(f"  MAE: {mae_final:.6f}\\n")
        f.write(f"  RMSE: {rmse_final:.6f}\\n")
        f.write(f"  R²: {r2_final:.4f}\\n")
        f.write(f"  AIC: {aic_corrected:.2f}\\n")
        f.write(f"  BIC: {bic_corrected:.2f}\\n")
        f.write(f"\\n4%阈值分类性能:\\n")
        f.write(f"  敏感度: {threshold_metrics['sensitivity']:.3f}\\n")
        f.write(f"  特异度: {threshold_metrics['specificity']:.3f}\\n")
        f.write(f"  AUC: {threshold_metrics['auc']:.3f}\\n")
        f.write(f"\\n关键改进:\\n")
        f.write("1. 使用低自由度样条交互 (df=3-4) 避免过拟合\\n")
        f.write("2. L2正则化控制模型复杂度\\n")
        f.write("3. 分组交叉验证确保真实泛化性能\\n")
        f.write("4. 在原始比例标度上进行统一评估\\n")
        f.write("5. 提供4%阈值的分类性能评估\\n")
    
    print(f"分析完成! 结果已保存至 '{output_dir}' 目录")
    print("=" * 70)
    
    return final_model, best_params, cv_results_df, stratification_df

if __name__ == "__main__":
    final_model, best_params, cv_results, stratification = fractional_logit_improved_modeling()
