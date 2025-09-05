#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一评估标度的GAM vs 分数logit回归对比分析
在同一比例标度上进行公平比较
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pygam import LinearGAM, s, te, l
import statsmodels.api as sm
from patsy import dmatrix

# 确保可以导入项目根目录的模块
sys.path.append('/Users/Mac/Downloads/mm')
from set_chinese_font import set_chinese_font

# 设置matplotlib以正确显示中文
set_chinese_font()
warnings.filterwarnings('ignore')

def create_interaction_features(data, df_gest=3, df_bmi=3, max_interactions=9):
    """创建孕周×BMI低自由度样条交互特征"""
    
    gest_splines = dmatrix(f"bs(孕周_标准化, df={df_gest}, degree=3) - 1", 
                          data, return_type='dataframe')
    bmi_splines = dmatrix(f"bs(BMI_标准化, df={df_bmi}, degree=3) - 1", 
                         data, return_type='dataframe')
    
    gest_splines.columns = [f'孕周_B{i+1}' for i in range(gest_splines.shape[1])]
    bmi_splines.columns = [f'BMI_B{i+1}' for i in range(bmi_splines.shape[1])]
    
    # 创建交互项
    interaction_features = []
    interaction_names = []
    
    count = 0
    for i in range(min(df_gest, gest_splines.shape[1])):
        for j in range(min(df_bmi, bmi_splines.shape[1])):
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

def evaluate_models_unified_scale():
    """在统一比例标度上评估和对比不同模型"""
    
    # 创建输出目录
    output_dir = 'unified_scale_comparison_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 70)
    print("统一标度模型对比 - 在比例标度上公平比较GAM vs 分数logit")
    print("=" * 70)
    
    # 加载处理后的数据 (现在是比例标度)
    try:
        data = pd.read_csv('processed_data.csv')
        print(f"加载数据样本数: {len(data)}")
    except FileNotFoundError:
        print("错误: 未找到 'processed_data.csv'")
        return
    
    # 准备建模数据
    y = data['Y染色体浓度'].copy()  # 现在是比例标度
    groups = data[data.columns[0]]  # 孕妇代码
    
    print(f"目标变量(比例标度): min={y.min():.6f}, max={y.max():.6f}, mean={y.mean():.6f}")
    print(f"4%阈值达标比例: {(y >= 0.04).mean():.3f}")
    
    # 核心特征
    core_vars = ['孕周_标准化', 'BMI_标准化', '年龄_标准化']
    quality_vars = ['总读段数_标准化', '比对比例_标准化', '重复读段比例_标准化']
    
    # 清理数据
    model_data = data[['Y染色体浓度', data.columns[0]] + 
                     core_vars + quality_vars].copy()
    model_data = model_data.dropna()
    
    y = model_data['Y染色体浓度'].copy()
    groups = model_data[data.columns[0]]
    
    print(f"清理后建模数据样本数: {len(model_data)}")
    
    # 模型定义和评估
    models_results = []
    
    # 1. GAM模型 (在比例标度上，使用Gaussian族)
    print("\n1. 评估GAM模型 (LinearGAM, Gaussian族, 比例标度)...")
    X_gam = model_data[core_vars].copy()
    
    try:
        # GAM模型配置
        gam_models = {
            'GAM_基准': LinearGAM(s(0) + s(1) + s(2)),
            'GAM_交互_WB': LinearGAM(s(0) + s(1) + s(2) + te(0, 1)),
            'GAM_简化': LinearGAM(s(0) + s(1) + l(2))
        }
        
        for model_name, gam_model in gam_models.items():
            print(f"  训练 {model_name}...")
            
            # 分组交叉验证
            group_kfold = GroupKFold(n_splits=5)
            cv_scores = {'mae': [], 'rmse': [], 'r2': []}
            
            for train_idx, test_idx in group_kfold.split(model_data, y, groups):
                X_train = X_gam.iloc[train_idx]
                X_test = X_gam.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                # 训练GAM
                gam_fit = gam_model.fit(X_train, y_train)
                y_pred = gam_fit.predict(X_test)
                
                # 计算指标
                cv_scores['mae'].append(mean_absolute_error(y_test, y_pred))
                cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                cv_scores['r2'].append(r2_score(y_test, y_pred))
            
            # 训练最终模型
            final_gam = gam_model.fit(X_gam, y)
            y_pred_final = final_gam.predict(X_gam)
            
            models_results.append({
                '模型类型': 'GAM',
                '模型名称': model_name,
                '标度': '比例标度',
                'CV_MAE_mean': np.mean(cv_scores['mae']),
                'CV_MAE_std': np.std(cv_scores['mae']),
                'CV_RMSE_mean': np.mean(cv_scores['rmse']),
                'CV_RMSE_std': np.std(cv_scores['rmse']),
                'CV_R2_mean': np.mean(cv_scores['r2']),
                'CV_R2_std': np.std(cv_scores['r2']),
                'Train_MAE': mean_absolute_error(y, y_pred_final),
                'Train_RMSE': np.sqrt(mean_squared_error(y, y_pred_final)),
                'Train_R2': r2_score(y, y_pred_final),
                '伪R2': final_gam.statistics_['pseudo_r2']['explained_deviance'],
                'AIC': final_gam.statistics_['AICc']
            })
            
            print(f"    CV RMSE: {np.mean(cv_scores['rmse']):.6f} ± {np.std(cv_scores['rmse']):.6f}")
            print(f"    CV R²: {np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}")
    
    except Exception as e:
        print(f"  GAM评估出错: {e}")
    
    # 2. 分数logit回归模型 (在比例标度上)
    print("\n2. 评估分数logit回归模型...")
    
    try:
        # 基础线性模型
        print("  训练基础线性分数logit...")
        X_linear = model_data[core_vars + quality_vars]
        
        group_kfold = GroupKFold(n_splits=5)
        cv_scores = {'mae': [], 'rmse': [], 'r2': []}
        
        for train_idx, test_idx in group_kfold.split(model_data, y, groups):
            X_train = X_linear.iloc[train_idx]
            X_test = X_linear.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # 训练GLM Binomial
            glm = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Binomial())
            model = glm.fit()
            y_pred = model.predict(sm.add_constant(X_test))
            
            cv_scores['mae'].append(mean_absolute_error(y_test, y_pred))
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            cv_scores['r2'].append(r2_score(y_test, y_pred))
        
        # 最终模型
        final_glm = sm.GLM(y, sm.add_constant(X_linear), family=sm.families.Binomial())
        final_model = final_glm.fit()
        y_pred_final = final_model.predict(sm.add_constant(X_linear))
        
        # 计算伪R2
        ll_null = sm.GLM(y, sm.add_constant(np.ones(len(y))), family=sm.families.Binomial()).fit().llf
        pseudo_r2 = 1 - (final_model.llf / ll_null)
        
        models_results.append({
            '模型类型': '分数logit',
            '模型名称': '基础线性',
            '标度': '比例标度',
            'CV_MAE_mean': np.mean(cv_scores['mae']),
            'CV_MAE_std': np.std(cv_scores['mae']),
            'CV_RMSE_mean': np.mean(cv_scores['rmse']),
            'CV_RMSE_std': np.std(cv_scores['rmse']),
            'CV_R2_mean': np.mean(cv_scores['r2']),
            'CV_R2_std': np.std(cv_scores['r2']),
            'Train_MAE': mean_absolute_error(y, y_pred_final),
            'Train_RMSE': np.sqrt(mean_squared_error(y, y_pred_final)),
            'Train_R2': r2_score(y, y_pred_final),
            '伪R2': pseudo_r2,
            'AIC': final_model.aic
        })
        
        print(f"    CV RMSE: {np.mean(cv_scores['rmse']):.6f} ± {np.std(cv_scores['rmse']):.6f}")
        print(f"    CV R²: {np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}")
        
        # 样条交互模型
        print("  训练样条交互分数logit...")
        X_spline = create_interaction_features(model_data, df_gest=3, df_bmi=3, max_interactions=9)
        X_spline_full = pd.concat([X_spline, model_data[['年龄_标准化'] + quality_vars]], axis=1)
        
        cv_scores = {'mae': [], 'rmse': [], 'r2': []}
        
        for train_idx, test_idx in group_kfold.split(model_data, y, groups):
            X_train = X_spline_full.iloc[train_idx]
            X_test = X_spline_full.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            glm = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Binomial())
            model = glm.fit()
            y_pred = model.predict(sm.add_constant(X_test))
            
            cv_scores['mae'].append(mean_absolute_error(y_test, y_pred))
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            cv_scores['r2'].append(r2_score(y_test, y_pred))
        
        final_glm_spline = sm.GLM(y, sm.add_constant(X_spline_full), family=sm.families.Binomial())
        final_model_spline = final_glm_spline.fit()
        y_pred_final_spline = final_model_spline.predict(sm.add_constant(X_spline_full))
        
        pseudo_r2_spline = 1 - (final_model_spline.llf / ll_null)
        
        models_results.append({
            '模型类型': '分数logit',
            '模型名称': '样条交互',
            '标度': '比例标度',
            'CV_MAE_mean': np.mean(cv_scores['mae']),
            'CV_MAE_std': np.std(cv_scores['mae']),
            'CV_RMSE_mean': np.mean(cv_scores['rmse']),
            'CV_RMSE_std': np.std(cv_scores['rmse']),
            'CV_R2_mean': np.mean(cv_scores['r2']),
            'CV_R2_std': np.std(cv_scores['r2']),
            'Train_MAE': mean_absolute_error(y, y_pred_final_spline),
            'Train_RMSE': np.sqrt(mean_squared_error(y, y_pred_final_spline)),
            'Train_R2': r2_score(y, y_pred_final_spline),
            '伪R2': pseudo_r2_spline,
            'AIC': final_model_spline.aic
        })
        
        print(f"    CV RMSE: {np.mean(cv_scores['rmse']):.6f} ± {np.std(cv_scores['rmse']):.6f}")
        print(f"    CV R²: {np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}")
    
    except Exception as e:
        print(f"  分数logit评估出错: {e}")
    
    # 结果汇总和保存
    print("\n3. 统一标度对比结果:")
    results_df = pd.DataFrame(models_results)
    print(results_df[['模型类型', '模型名称', 'CV_RMSE_mean', 'CV_RMSE_std', 'CV_R2_mean', 'CV_R2_std', '伪R2']].to_string(index=False))
    
    # 保存结果
    results_df.to_csv(f'{output_dir}/unified_comparison_results.csv', 
                     index=False, encoding='utf-8-sig')
    
    # 创建BMI分层分析（修复版本）
    print("\n4. 创建BMI分层分析...")
    create_fixed_bmi_stratification(model_data, final_model, output_dir)
    
    # 创建详细报告
    create_comparison_report(results_df, output_dir)
    
    print(f"\n统一标度对比分析完成! 结果已保存至 '{output_dir}' 目录")
    print("=" * 70)
    
    return results_df

def create_fixed_bmi_stratification(data, model, output_dir):
    """创建修复的BMI分层分析"""
    
    try:
        # 从标准化数据还原BMI
        bmi_col_idx = list(data.columns).index('BMI_标准化')
        bmi_std_data = data['BMI_标准化']
        
        # 计算原始BMI (需要从原始数据获取均值和标准差)
        original_data = pd.read_excel('/Users/Mac/Downloads/mm/附件.xlsx', sheet_name='男胎检测数据')
        bmi_original = pd.to_numeric(original_data.iloc[:, 10], errors='coerce')  # K列
        bmi_mean = bmi_original.mean()
        bmi_std = bmi_original.std()
        
        # 还原BMI
        bmi_raw = bmi_std_data * bmi_std + bmi_mean
        
        # 从标准化数据还原孕周
        gest_col_idx = list(data.columns).index('孕周_标准化')
        gest_std_data = data['孕周_标准化']
        
        # 需要从processed_data中获取孕周的均值和标准差
        processed_data = pd.read_csv('processed_data.csv')
        if '孕周_小数' in processed_data.columns:
            gest_original = processed_data['孕周_小数'].dropna()
            gest_mean = gest_original.mean()
            gest_std = gest_original.std()
            
            gest_raw = gest_std_data * gest_std + gest_mean
        else:
            gest_raw = gest_std_data * 2 + 15  # 粗略估计
        
        # BMI分组
        bmi_bins = [20, 28, 32, 36, 40, 100]
        bmi_labels = ['[20,28)', '[28,32)', '[32,36)', '[36,40)', '[40,100)']
        bmi_groups = pd.cut(bmi_raw, bins=bmi_bins, labels=bmi_labels, right=False, include_lowest=True)
        
        # 分层分析
        stratification_results = []
        
        for group_name in bmi_labels:
            group_mask = (bmi_groups == group_name)
            group_count = group_mask.sum()
            
            if group_count < 5:  # 样本量太小跳过
                continue
            
            group_data = data[group_mask].copy()
            group_bmi = bmi_raw[group_mask]
            group_gest = gest_raw[group_mask]
            group_y = data.loc[group_mask, 'Y染色体浓度']
            
            # 创建预测用的特征矩阵
            core_vars = ['孕周_标准化', 'BMI_标准化', '年龄_标准化']
            quality_vars = ['总读段数_标准化', '比对比例_标准化', '重复读段比例_标准化']
            X_group = group_data[core_vars + quality_vars]
            
            try:
                # 预测
                y_pred_group = model.predict(sm.add_constant(X_group))
                
                # 达标分析
                达标mask = (y_pred_group >= 0.04)
                if 达标mask.sum() > 0:
                    最早达标孕周 = group_gest[达标mask].min()
                    达标比例 = 达标mask.mean()
                else:
                    最早达标孕周 = np.nan
                    达标比例 = 0.0
                
                # 真实达标分析
                真实达标mask = (group_y >= 0.04)
                真实达标比例 = 真实达标mask.mean()
                if 真实达标mask.sum() > 0:
                    真实最早达标孕周 = group_gest[真实达标mask].min()
                else:
                    真实最早达标孕周 = np.nan
                
                stratification_results.append({
                    'BMI组': group_name,
                    '样本数': group_count,
                    'BMI均值': group_bmi.mean(),
                    'BMI范围': f"{group_bmi.min():.1f}-{group_bmi.max():.1f}",
                    '孕周范围': f"{group_gest.min():.1f}-{group_gest.max():.1f}",
                    '预测最早达标孕周': 最早达标孕周,
                    '预测达标比例': 达标比例,
                    '真实最早达标孕周': 真实最早达标孕周,
                    '真实达标比例': 真实达标比例,
                    'Y浓度均值': group_y.mean(),
                    'Y浓度标准差': group_y.std()
                })
                
            except Exception as e:
                print(f"BMI组 {group_name} 分析失败: {e}")
                continue
        
        # 保存结果
        if stratification_results:
            stratification_df = pd.DataFrame(stratification_results)
            stratification_df.to_csv(f'{output_dir}/bmi_risk_stratification_fixed.csv', 
                                   index=False, encoding='utf-8-sig')
            
            print("BMI分层结果:")
            for _, row in stratification_df.iterrows():
                if not pd.isna(row['预测最早达标孕周']):
                    print(f"  {row['BMI组']}: 预测{row['预测最早达标孕周']:.1f}周, 真实{row['真实最早达标孕周']:.1f}周")
                else:
                    print(f"  {row['BMI组']}: 样本不足或无达标")
            
            return stratification_df
        else:
            print("BMI分层分析失败: 无有效结果")
            return None
    
    except Exception as e:
        print(f"BMI分层分析出错: {e}")
        return None

def create_comparison_report(results_df, output_dir):
    """创建统一标度对比报告"""
    
    with open(f'{output_dir}/unified_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("统一标度模型对比分析报告\\n")
        f.write("=" * 70 + "\\n\\n")
        
        f.write("对比说明:\\n")
        f.write("本报告在统一的比例标度上对比GAM和分数logit回归模型\\n")
        f.write("避免了跨标度比较的误导性结论\\n\\n")
        
        f.write("模型性能对比 (基于5折分组交叉验证):\\n")
        f.write("=" * 50 + "\\n")
        
        # 按CV RMSE排序
        sorted_results = results_df.sort_values('CV_RMSE_mean')
        
        for _, row in sorted_results.iterrows():
            f.write(f"\\n{row['模型类型']} - {row['模型名称']}:\\n")
            f.write(f"  CV RMSE: {row['CV_RMSE_mean']:.6f} ± {row['CV_RMSE_std']:.6f}\\n")
            f.write(f"  CV R²: {row['CV_R2_mean']:.4f} ± {row['CV_R2_std']:.4f}\\n")
            f.write(f"  CV MAE: {row['CV_MAE_mean']:.6f} ± {row['CV_MAE_std']:.6f}\\n")
            f.write(f"  伪R²: {row['伪R2']:.4f}\\n")
            f.write(f"  AIC: {row['AIC']:.2f}\\n")
        
        f.write("\\n\\n关键结论:\\n")
        f.write("=" * 30 + "\\n")
        
        best_model = sorted_results.iloc[0]
        f.write(f"1. 最佳模型: {best_model['模型类型']} - {best_model['模型名称']}\\n")
        f.write(f"2. 最佳CV RMSE: {best_model['CV_RMSE_mean']:.6f}\\n")
        f.write(f"3. 在统一比例标度上，模型间的真实性能差异得到了客观评估\\n")
        
        # 对比GAM和分数logit的平均性能
        gam_results = results_df[results_df['模型类型'] == 'GAM']
        logit_results = results_df[results_df['模型类型'] == '分数logit']
        
        if not gam_results.empty and not logit_results.empty:
            gam_avg_rmse = gam_results['CV_RMSE_mean'].mean()
            logit_avg_rmse = logit_results['CV_RMSE_mean'].mean()
            
            f.write(f"4. GAM平均CV RMSE: {gam_avg_rmse:.6f}\\n")
            f.write(f"5. 分数logit平均CV RMSE: {logit_avg_rmse:.6f}\\n")
            
            if logit_avg_rmse < gam_avg_rmse:
                improvement = (gam_avg_rmse - logit_avg_rmse) / gam_avg_rmse * 100
                f.write(f"6. 分数logit相对GAM改进: {improvement:.1f}%\\n")
            else:
                improvement = (logit_avg_rmse - gam_avg_rmse) / logit_avg_rmse * 100
                f.write(f"6. GAM相对分数logit改进: {improvement:.1f}%\\n")

if __name__ == "__main__":
    results = evaluate_models_unified_scale()
