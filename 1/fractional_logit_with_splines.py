#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于分数logit回归的Y染色体浓度建模
使用GLM Binomial族 + logit链接 + 样条基函数进行非线性建模
纳入测序质量变量，使用按孕妇代码分组的交叉验证
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from patsy import dmatrix
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import itertools

# 确保可以导入项目根目录的模块
sys.path.append('/Users/Mac/Downloads/mm')
from set_chinese_font import set_chinese_font

# 设置matplotlib以正确显示中文
set_chinese_font()

# 忽略一些警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def create_spline_features(data, var_name, df=5, degree=3):
    """
    为指定变量创建B样条基函数特征
    
    Parameters:
    -----------
    data : array-like
        输入变量数据
    var_name : str
        变量名称
    df : int
        自由度（样条基函数的数量）
    degree : int
        样条度数
        
    Returns:
    --------
    spline_matrix : ndarray
        样条基函数矩阵
    """
    # 使用patsy创建B样条基
    formula = f"bs({var_name}, df={df}, degree={degree})"
    spline_data = pd.DataFrame({var_name: data})
    spline_matrix = dmatrix(formula, spline_data, return_type='dataframe')
    
    return spline_matrix

def fractional_logit_modeling():
    """
    使用分数logit回归对Y染色体浓度进行建模
    """
    
    # 1. 创建输出目录
    output_dir = 'fractional_logit_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 70)
    print("分数logit回归建模 - Y染色体浓度分析")
    print("=" * 70)
    
    # 2. 重新运行数据预处理以获得正确的比例数据
    print("\n1. 重新处理数据以获得正确的比例格式...")
    
    # 先运行修改后的数据处理
    from data_process import process_data
    process_data()
    
    # 加载处理后的数据
    try:
        data = pd.read_csv('processed_data.csv')
        print(f"加载的数据样本数: {len(data)}")
    except FileNotFoundError:
        print("错误: 未找到 'processed_data.csv'。")
        return
    
    # 3. 准备建模数据
    print("\n2. 准备建模数据...")
    
    # 目标变量（现在是正确的比例数据）
    y = data['Y染色体浓度'].copy()
    print(f"目标变量统计: min={y.min():.6f}, max={y.max():.6f}, mean={y.mean():.6f}")
    print(f"4%阈值达标比例: {(y >= 0.04).mean():.3f}")
    
    # 孕妇代码用于分组交叉验证
    groups = data[data.columns[0]]  # 孕妇代码
    
    # 核心预测变量
    core_vars = ['孕周_标准化', 'BMI_标准化', '年龄_标准化']
    
    # 测序质量变量
    quality_vars = [
        '总读段数_标准化', '比对比例_标准化', '重复读段比例_标准化',
        '唯一比对读段比例_标准化', 'GC含量_标准化', '过滤读段比例_标准化'
    ]
    
    # 其他协变量
    other_vars = ['怀孕次数_标准化', '生产次数_标准化']
    
    # 检查可用变量
    available_vars = []
    for var_list, var_type in [(core_vars, '核心变量'), (quality_vars, '测序质量变量'), (other_vars, '其他变量')]:
        available_in_group = [var for var in var_list if var in data.columns]
        available_vars.extend(available_in_group)
        print(f"{var_type}: {len(available_in_group)}/{len(var_list)} 个可用")
        for var in available_in_group:
            missing_rate = data[var].isna().mean()
            print(f"  - {var}: 缺失率 {missing_rate:.1%}")
    
    # 清理数据
    model_data = data[['Y染色体浓度'] + [data.columns[0]] + available_vars].copy()
    model_data = model_data.dropna()
    print(f"清理后的建模数据样本数: {len(model_data)}")
    
    # 更新变量
    y = model_data['Y染色体浓度'].copy()
    groups = model_data[data.columns[0]]
    X_vars = [var for var in available_vars if var in model_data.columns]
    
    # 4. 构建不同复杂度的模型
    print(f"\n3. 构建分数logit回归模型...")
    
    models = {}
    model_results = []
    
    # 模型1: 核心变量 + 样条
    print("  构建模型1: 核心变量(孕周、BMI、年龄) + 样条...")
    core_available = [var for var in core_vars if var in model_data.columns]
    if len(core_available) >= 2:
        try:
            # 为孕周和BMI创建样条基
            gestational_splines = create_spline_features(
                model_data['孕周_标准化'], '孕周_标准化', df=5)
            bmi_splines = create_spline_features(
                model_data['BMI_标准化'], 'BMI_标准化', df=5)
            
            # 构建设计矩阵
            X1 = pd.concat([
                gestational_splines,
                bmi_splines,
                model_data[['年龄_标准化']] if '年龄_标准化' in model_data.columns else pd.DataFrame()
            ], axis=1)
            
            # 拟合GLM
            glm1 = sm.GLM(y, sm.add_constant(X1), family=sm.families.Binomial())
            models['M1_core_splines'] = glm1.fit()
            print(f"    模型1拟合完成: {X1.shape[1]} 个特征")
            
        except Exception as e:
            print(f"    模型1拟合失败: {e}")
    
    # 模型2: 核心变量 + 测序质量变量 + 样条
    print("  构建模型2: 核心变量 + 测序质量变量 + 样条...")
    quality_available = [var for var in quality_vars if var in model_data.columns]
    if len(core_available) >= 2 and len(quality_available) >= 3:
        try:
            # 构建设计矩阵
            X2 = pd.concat([
                gestational_splines,
                bmi_splines,
                model_data[['年龄_标准化']] if '年龄_标准化' in model_data.columns else pd.DataFrame(),
                model_data[quality_available]
            ], axis=1)
            
            # 拟合GLM
            glm2 = sm.GLM(y, sm.add_constant(X2), family=sm.families.Binomial())
            models['M2_core_quality_splines'] = glm2.fit()
            print(f"    模型2拟合完成: {X2.shape[1]} 个特征")
            
        except Exception as e:
            print(f"    模型2拟合失败: {e}")
    
    # 模型3: 全变量 + 样条
    print("  构建模型3: 全变量 + 样条...")
    other_available = [var for var in other_vars if var in model_data.columns]
    if len(core_available) >= 2:
        try:
            # 构建设计矩阵
            X3_parts = [gestational_splines, bmi_splines]
            if '年龄_标准化' in model_data.columns:
                X3_parts.append(model_data[['年龄_标准化']])
            if quality_available:
                X3_parts.append(model_data[quality_available])
            if other_available:
                X3_parts.append(model_data[other_available])
                
            X3 = pd.concat(X3_parts, axis=1)
            
            # 拟合GLM
            glm3 = sm.GLM(y, sm.add_constant(X3), family=sm.families.Binomial())
            models['M3_full_splines'] = glm3.fit()
            print(f"    模型3拟合完成: {X3.shape[1]} 个特征")
            
        except Exception as e:
            print(f"    模型3拟合失败: {e}")
    
    # 模型4: 简化线性模型（作为基准）
    print("  构建模型4: 简化线性基准模型...")
    if len(core_available) >= 2:
        try:
            X4 = model_data[core_available + quality_available[:3]]  # 只取前3个质量变量
            glm4 = sm.GLM(y, sm.add_constant(X4), family=sm.families.Binomial())
            models['M4_linear_baseline'] = glm4.fit()
            print(f"    模型4拟合完成: {X4.shape[1]} 个特征")
            
        except Exception as e:
            print(f"    模型4拟合失败: {e}")
    
    # 5. 模型比较
    print(f"\n4. 模型性能比较...")
    
    if not models:
        print("错误: 没有成功拟合的模型")
        return
    
    comparison_results = []
    for name, model in models.items():
        # 计算拟合指标
        y_pred = model.fittedvalues
        
        # 在原始比例标度上计算指标
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # McFadden Pseudo R²
        ll_null = sm.GLM(y, sm.add_constant(np.ones(len(y))), family=sm.families.Binomial()).fit().llf
        ll_model = model.llf
        pseudo_r2 = 1 - (ll_model / ll_null)
        
        comparison_results.append({
            '模型': name,
            'AIC': f"{model.aic:.2f}",
            'BIC': f"{model.bic:.2f}",
            'Log-Likelihood': f"{model.llf:.2f}",
            'Pseudo R²': f"{pseudo_r2:.4f}",
            'MAE': f"{mae:.6f}",
            'RMSE': f"{rmse:.6f}",
            '参数数量': len(model.params)
        })
    
    results_df = pd.DataFrame(comparison_results)
    print("\n模型比较结果:")
    print(results_df.to_string(index=False))
    
    # 选择最优模型（基于AIC）
    best_model_name = min(models.keys(), key=lambda k: models[k].aic)
    best_model = models[best_model_name]
    print(f"\n最优模型 (基于AIC): {best_model_name}")
    
    # 6. 分组交叉验证
    print(f"\n5. 进行按孕妇代码分组的5折交叉验证...")
    cv_results = perform_grouped_cv(model_data, y, groups, core_available, quality_available)
    
    print("\n分组交叉验证结果:")
    cv_df = pd.DataFrame(cv_results)
    print(cv_df.to_string(index=False))
    
    # 7. 最优模型详细分析
    print(f"\n6. 最优模型 ({best_model_name}) 详细分析...")
    print("\n模型摘要:")
    print(best_model.summary())
    
    # 8. 可视化分析
    print(f"\n7. 生成可视化分析...")
    create_diagnostic_plots(best_model, y, model_data, core_available, output_dir, best_model_name)
    
    # 9. 保存结果
    print(f"\n8. 保存分析结果...")
    
    # 保存模型比较结果
    results_df.to_csv(f'{output_dir}/model_comparison.csv', index=False, encoding='utf-8-sig')
    
    # 保存交叉验证结果
    cv_df.to_csv(f'{output_dir}/cv_results.csv', index=False, encoding='utf-8-sig')
    
    # 保存最优模型摘要
    with open(f'{output_dir}/best_model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"分数logit回归最优模型: {best_model_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write("模型比较:\n")
        f.write(results_df.to_string(index=False) + "\n\n")
        f.write("=" * 70 + "\n")
        f.write("分组交叉验证结果:\n")
        f.write(cv_df.to_string(index=False) + "\n\n")
        f.write("=" * 70 + "\n")
        f.write("最优模型详细摘要:\n")
        f.write(str(best_model.summary()) + "\n\n")
        f.write("=" * 70 + "\n")
        f.write("关键发现:\n")
        f.write("1. 使用分数logit回归（GLM Binomial + logit链接）处理比例响应变量\n")
        f.write("2. 保持Y染色体浓度的原始比例含义（4%阈值直接可解释）\n") 
        f.write("3. 纳入测序质量变量控制测量噪声和系统性偏差\n")
        f.write("4. 使用样条基函数捕获非线性关系\n")
        f.write("5. 采用按孕妇代码分组的交叉验证避免数据泄漏\n")
    
    print(f"\n分析完成! 所有结果已保存至 '{output_dir}' 目录")
    print("=" * 70)

def perform_grouped_cv(data, y, groups, core_vars, quality_vars):
    """
    执行按孕妇代码分组的交叉验证
    """
    
    cv_results = []
    
    # 不同模型配置
    model_configs = [
        ('Core_Splines', core_vars, True),
        ('Core_Quality', core_vars + quality_vars[:3], False),
        ('Core_Quality_Splines', core_vars + quality_vars[:3], True),
    ]
    
    for config_name, vars_list, use_splines in model_configs:
        if not all(var in data.columns for var in vars_list):
            continue
            
        try:
            # 分组5折交叉验证
            group_kfold = GroupKFold(n_splits=5)
            mae_scores = []
            rmse_scores = []
            r2_scores = []
            
            for train_idx, test_idx in group_kfold.split(data, y, groups):
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                # 构建特征矩阵
                if use_splines and len([v for v in ['孕周_标准化', 'BMI_标准化'] if v in vars_list]) >= 2:
                    # 使用样条
                    train_splines_gest = create_spline_features(train_data['孕周_标准化'], '孕周_标准化', df=4)
                    train_splines_bmi = create_spline_features(train_data['BMI_标准化'], 'BMI_标准化', df=4)
                    
                    test_splines_gest = create_spline_features(test_data['孕周_标准化'], '孕周_标准化', df=4)
                    test_splines_bmi = create_spline_features(test_data['BMI_标准化'], 'BMI_标准化', df=4)
                    
                    other_vars = [v for v in vars_list if v not in ['孕周_标准化', 'BMI_标准化']]
                    
                    X_train = pd.concat([train_splines_gest, train_splines_bmi, train_data[other_vars]], axis=1)
                    X_test = pd.concat([test_splines_gest, test_splines_bmi, test_data[other_vars]], axis=1)
                else:
                    # 线性模型
                    X_train = train_data[vars_list]
                    X_test = test_data[vars_list]
                
                # 拟合模型
                glm = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Binomial())
                model = glm.fit()
                
                # 预测
                y_pred = model.predict(sm.add_constant(X_test))
                
                # 计算指标
                mae_scores.append(mean_absolute_error(y_test, y_pred))
                rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                r2_scores.append(r2_score(y_test, y_pred))
            
            cv_results.append({
                '模型配置': config_name,
                'MAE_mean': f"{np.mean(mae_scores):.6f}",
                'MAE_std': f"{np.std(mae_scores):.6f}",
                'RMSE_mean': f"{np.mean(rmse_scores):.6f}",
                'RMSE_std': f"{np.std(rmse_scores):.6f}",
                'R2_mean': f"{np.mean(r2_scores):.4f}",
                'R2_std': f"{np.std(r2_scores):.4f}"
            })
            
        except Exception as e:
            print(f"    交叉验证 {config_name} 失败: {e}")
    
    return cv_results

def create_diagnostic_plots(model, y_true, data, core_vars, output_dir, model_name):
    """
    创建模型诊断图
    """
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'分数Logit回归模型诊断 - {model_name}', fontsize=16, fontweight='bold')
        
        y_pred = model.fittedvalues
        residuals = y_true - y_pred
        
        # 1. 拟合值 vs 真实值
        axes[0,0].scatter(y_true, y_pred, alpha=0.6, color='blue', s=20)
        axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('真实值')
        axes[0,0].set_ylabel('预测值')
        axes[0,0].set_title('预测值 vs 真实值')
        
        # 计算相关系数
        corr = np.corrcoef(y_true, y_pred)[0,1]
        axes[0,0].text(0.05, 0.95, f'相关系数: {corr:.4f}', transform=axes[0,0].transAxes, 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. 残差 vs 拟合值
        axes[0,1].scatter(y_pred, residuals, alpha=0.6, color='green', s=20)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('拟合值')
        axes[0,1].set_ylabel('残差')
        axes[0,1].set_title('残差 vs 拟合值')
        
        # 3. 残差直方图
        axes[0,2].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[0,2].set_xlabel('残差')
        axes[0,2].set_ylabel('频数')
        axes[0,2].set_title('残差分布')
        
        # 添加正态性统计
        _, p_value = stats.normaltest(residuals)
        axes[0,2].text(0.05, 0.95, f'正态性检验 p值: {p_value:.4f}', transform=axes[0,2].transAxes,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Q-Q图
        stats.probplot(residuals, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q图 (残差正态性)')
        
        # 5. 杠杆值分析
        if hasattr(model, 'get_influence'):
            influence = model.get_influence()
            axes[1,1].scatter(range(len(influence.hat_matrix_diag)), influence.hat_matrix_diag, 
                            alpha=0.6, color='orange', s=20)
            axes[1,1].set_xlabel('观测值索引')
            axes[1,1].set_ylabel('杠杆值')
            axes[1,1].set_title('杠杆值分析')
            
            # 标记高杠杆值点
            threshold = 2 * len(model.params) / len(y_true)
            high_leverage = influence.hat_matrix_diag > threshold
            if high_leverage.sum() > 0:
                axes[1,1].axhline(y=threshold, color='r', linestyle='--', 
                                label=f'阈值: {threshold:.3f}')
                axes[1,1].legend()
        else:
            axes[1,1].text(0.5, 0.5, '杠杆值信息不可用', ha='center', va='center', 
                          transform=axes[1,1].transAxes, fontsize=12)
            axes[1,1].set_title('杠杆值分析')
        
        # 6. 特征重要性（回归系数）
        if hasattr(model, 'params'):
            # 获取非常数项的系数
            params = model.params[1:]  # 排除截距
            param_names = [str(name)[:15] + '...' if len(str(name)) > 15 else str(name) 
                          for name in model.params.index[1:]]
            
            # 选择前15个最重要的系数
            abs_params = np.abs(params)
            top_indices = abs_params.nlargest(15).index
            top_params = params[top_indices]
            top_names = [param_names[list(params.index).index(idx)] for idx in top_indices]
            
            y_pos = np.arange(len(top_params))
            colors = ['red' if x < 0 else 'blue' for x in top_params]
            
            axes[1,2].barh(y_pos, top_params, color=colors, alpha=0.7)
            axes[1,2].set_yticks(y_pos)
            axes[1,2].set_yticklabels(top_names, fontsize=8)
            axes[1,2].set_xlabel('系数值')
            axes[1,2].set_title('Top 15 特征系数')
            axes[1,2].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/diagnostic_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  诊断图已保存: {output_dir}/diagnostic_plots.png")
        
        # 额外生成样条效应图（如果适用）
        create_spline_effects_plot(data, core_vars, output_dir)
        
    except Exception as e:
        print(f"  创建诊断图时出错: {e}")

def create_spline_effects_plot(data, core_vars, output_dir):
    """
    创建样条效应图
    """
    
    try:
        if '孕周_标准化' in core_vars and 'BMI_标准化' in core_vars:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('样条基函数效应分析', fontsize=14, fontweight='bold')
            
            # 孕周样条基可视化
            gestational_range = np.linspace(data['孕周_标准化'].min(), data['孕周_标准化'].max(), 100)
            gestational_splines = create_spline_features(gestational_range, '孕周_标准化', df=5)
            
            for i, col in enumerate(gestational_splines.columns[:5]):  # 显示前5个基函数
                axes[0].plot(gestational_range, gestational_splines[col], 
                           label=f'基函数 {i+1}', alpha=0.8)
            
            axes[0].set_xlabel('孕周 (标准化)')
            axes[0].set_ylabel('样条基函数值')
            axes[0].set_title('孕周样条基函数')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # BMI样条基可视化
            bmi_range = np.linspace(data['BMI_标准化'].min(), data['BMI_标准化'].max(), 100)
            bmi_splines = create_spline_features(bmi_range, 'BMI_标准化', df=5)
            
            for i, col in enumerate(bmi_splines.columns[:5]):  # 显示前5个基函数
                axes[1].plot(bmi_range, bmi_splines[col], 
                           label=f'基函数 {i+1}', alpha=0.8)
            
            axes[1].set_xlabel('BMI (标准化)')
            axes[1].set_ylabel('样条基函数值')
            axes[1].set_title('BMI样条基函数')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/spline_basis_functions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  样条基函数图已保存: {output_dir}/spline_basis_functions.png")
            
    except Exception as e:
        print(f"  创建样条效应图时出错: {e}")

if __name__ == "__main__":
    fractional_logit_modeling()
