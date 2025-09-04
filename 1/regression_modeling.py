#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y染色体浓度关系建模分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.graphics.gofplots import qqplot
import os
import sys
sys.path.append('/Users/Mac/Downloads/mm')
from set_chinese_font import set_chinese_font

# 设置中文字体
set_chinese_font()

def regression_modeling():
    """Y染色体浓度关系建模"""
    
    # 创建输出目录
    output_dir = 'regression_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取数据
    df = pd.read_csv('processed_data.csv')
    
    # 重新读取原始数据并进行处理以获得孕周信息
    original_df = pd.read_excel('/Users/Mac/Downloads/mm/附件.xlsx', sheet_name='男胎检测数据')
    
    # 数据清洗 - 与data_process.py保持一致
    ae_col = original_df.columns[-1]  # 胎儿是否健康
    original_df = original_df[original_df[ae_col] == '是'].copy()
    
    # 质量控制筛选 - 简化版本，与处理脚本保持一致的逻辑
    quality_cols = [11, 12, 13, 14, 15, 26]  # L, M, N, O, P, AA列
    for col_idx in quality_cols:
        col_data = pd.to_numeric(original_df.iloc[:, col_idx], errors='coerce')
        mean_val = col_data.mean()
        std_val = col_data.std()
        if col_idx in [11, 12, 14, 15]:  # L, M, O, P - 剔除低值
            threshold = mean_val - 3 * std_val
            original_df = original_df[col_data >= threshold].copy()
        else:  # N, AA - 剔除高值
            threshold = mean_val + 3 * std_val
            original_df = original_df[col_data <= threshold].copy()
    
    # 转换孕周格式
    def convert_gestational_week(week_str):
        if pd.isna(week_str):
            return np.nan
        try:
            week_str = str(week_str).strip()
            if 'w' in week_str:
                parts = week_str.split('w')
                weeks = float(parts[0])
                if '+' in parts[1]:
                    days = float(parts[1].replace('+', ''))
                    return weeks + days / 7.0
                else:
                    return weeks
            else:
                return float(week_str)
        except:
            return np.nan
    
    original_df['孕周_小数'] = original_df.iloc[:, 9].apply(convert_gestational_week)
    
    # 筛选10-25周的样本
    df_filtered = original_df[(original_df['孕周_小数'] >= 10) & (original_df['孕周_小数'] <= 25)].copy()
    
    print(f"筛选10-25周样本: {len(df_filtered)} 个 (清洗后总样本: {len(original_df)} 个)")
    
    # 从筛选后的数据中提取和标准化变量
    y = df_filtered.iloc[:, 21]  # V列 - Y染色体浓度
    
    # 处理怀孕次数特殊值
    def clean_pregnancy_count(value):
        if pd.isna(value):
            return np.nan
        value_str = str(value).strip()
        if '≥' in value_str or '>=' in value_str:
            import re
            numbers = re.findall(r'\d+', value_str)
            if numbers:
                return float(numbers[0])
        try:
            return float(value_str)
        except:
            return np.nan
    
    # 提取原始变量
    gestational_week = df_filtered['孕周_小数']
    bmi = pd.to_numeric(df_filtered.iloc[:, 10], errors='coerce')  # K列
    age = pd.to_numeric(df_filtered.iloc[:, 2], errors='coerce')   # C列
    pregnancy_count = df_filtered.iloc[:, 28].apply(clean_pregnancy_count)  # AC列
    birth_count = pd.to_numeric(df_filtered.iloc[:, 29], errors='coerce')   # AD列
    
    # Z-score标准化
    def standardize(x):
        return (x - x.mean()) / x.std()
    
    X_vars = {
        'W': standardize(gestational_week),    # 孕周
        'B': standardize(bmi),                 # BMI
        'A': standardize(age),                 # 年龄
        'C': standardize(pregnancy_count),     # 怀孕次数
        'D': standardize(birth_count)          # 生产次数
    }
    
    # 移除缺失值
    data = pd.DataFrame(X_vars)
    data['y'] = y
    data = data.dropna()
    
    print(f"有效样本数: {len(data)}")
    
    # 1. 基础线性回归模型
    print("\n1. 基础线性回归模型")
    print("=" * 50)
    
    X_basic = data[['W', 'B', 'A']]
    X_basic_sm = sm.add_constant(X_basic)
    
    model_basic = sm.OLS(data['y'], X_basic_sm).fit()
    
    print("基础模型结果:")
    print(f"R²: {model_basic.rsquared:.4f}")
    print(f"调整R²: {model_basic.rsquared_adj:.4f}")
    print(f"AIC: {model_basic.aic:.2f}")
    print(f"模型p值: {model_basic.f_pvalue:.6f}")
    
    print("\n回归系数:")
    for i, var in enumerate(['截距', '孕周(W)', 'BMI(B)', '年龄(A)']):
        coef = model_basic.params.iloc[i]
        pval = model_basic.pvalues.iloc[i]
        print(f"{var}: {coef:.6f} (p={pval:.6f})")
    
    # 2. 逐步回归筛选最优模型
    print("\n2. 逐步回归模型选择")
    print("=" * 50)
    
    # 构建候选项
    data_expanded = data.copy()
    
    # 添加二次项
    for var in ['W', 'B', 'A', 'C', 'D']:
        data_expanded[f'{var}2'] = data_expanded[var] ** 2
    
    # 添加交互项
    interactions = [('W', 'B'), ('W', 'A'), ('B', 'A'), ('W', 'C'), ('B', 'D')]
    for var1, var2 in interactions:
        data_expanded[f'{var1}x{var2}'] = data_expanded[var1] * data_expanded[var2]
    
    # 基础变量（始终在模型中）
    base_vars = ['W', 'B', 'A']
    # 仅将二次项与交互项作为候选
    candidate_vars = ['W2', 'B2', 'A2', 'C2', 'D2',
                      'WxB', 'WxA', 'BxA', 'WxC', 'BxD']
    
    # 逐步回归（前向，仅添加候选项）
    selected_vars = []
    current_aic = model_basic.aic
    selection_trace = []  # 记录逐步选择过程
    
    print("逐步回归过程:")
    
    while True:
        best_var = None
        best_aic = current_aic
        best_model = None
        
        for var in candidate_vars:
            if var not in selected_vars:
                test_vars = base_vars + selected_vars + [var]
                X_test = data_expanded[test_vars]
                X_test_sm = sm.add_constant(X_test)
                
                try:
                    model_test = sm.OLS(data_expanded['y'], X_test_sm).fit()
                    
                    # 检查新增变量的显著性
                    new_var_pval = model_test.pvalues[var]
                    
                    if model_test.aic < best_aic and new_var_pval < 0.05:
                        best_aic = model_test.aic
                        best_var = var
                        best_model = model_test
                except:
                    continue
        
        if best_var is not None:
            selected_vars.append(best_var)
            current_aic = best_aic
            new_var_pval = best_model.pvalues[best_var]
            print(f"添加项: {best_var} (AIC: {current_aic:.2f}, p值: {new_var_pval:.6f})")
            # 记录到选择轨迹
            selection_trace.append({
                '步骤': len(selected_vars),
                '新增变量': best_var,
                'AIC': round(best_model.aic, 4),
                'R²': round(best_model.rsquared, 6),
                '调整R²': round(best_model.rsquared_adj, 6),
                '新增变量p值': round(float(new_var_pval), 8)
            })
        else:
            break
    
    # 最终模型
    if selected_vars:
        X_final = data_expanded[base_vars + selected_vars]
        X_final_sm = sm.add_constant(X_final)
        model_final = sm.OLS(data_expanded['y'], X_final_sm).fit()
        
        print(f"\n最终模型包含变量(基础+新增): {base_vars + selected_vars}")
        print(f"最终模型AIC: {model_final.aic:.2f}")
        print(f"最终模型R²: {model_final.rsquared:.4f}")
        print(f"最终模型调整R²: {model_final.rsquared_adj:.4f}")
        
        print("\n最终模型系数:")
        for var in ['const'] + base_vars + selected_vars:
            coef = model_final.params[var]
            pval = model_final.pvalues[var]
            print(f"{var}: {coef:.6f} (p={pval:.6f})")
    else:
        print("未找到显著改善的变量")
        model_final = model_basic
    
    # 计算并保存VIF（多重共线性）
    feature_vars = base_vars + selected_vars
    
    X_for_vif = data_expanded[feature_vars].copy()
    vif_records = []
    for i, col in enumerate(X_for_vif.columns):
        try:
            vif_val = variance_inflation_factor(X_for_vif.values, i)
        except Exception:
            vif_val = np.nan
        vif_records.append({'变量': col, 'VIF': float(vif_val) if pd.notna(vif_val) else np.nan})
    vif_df = pd.DataFrame(vif_records)
    vif_df.sort_values('VIF', ascending=False, inplace=True)
    vif_df.to_csv(f'{output_dir}/vif.csv', index=False, encoding='utf-8-sig')
    
    # 诊断图：残差-拟合、QQ、尺度-位置、杠杆/Cook距离
    try:
        influence = OLSInfluence(model_final)
        fitted = model_final.fittedvalues
        resid = model_final.resid
        stud_resid = influence.resid_studentized_internal
        leverage = influence.hat_matrix_diag
        cooks = influence.cooks_distance[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # 残差 vs 拟合值
        axes[0, 0].scatter(fitted, resid, alpha=0.7)
        axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[0, 0].set_xlabel('拟合值')
        axes[0, 0].set_ylabel('残差')
        axes[0, 0].set_title('残差 vs 拟合值')
        
        # QQ图
        qqplot(stud_resid, line='45', ax=axes[0, 1])
        axes[0, 1].set_title('QQ图（学生化残差）')
        
        # 尺度-位置图
        axes[1, 0].scatter(fitted, np.sqrt(np.abs(stud_resid)), alpha=0.7)
        axes[1, 0].set_xlabel('拟合值')
        axes[1, 0].set_ylabel('sqrt(|学生化残差|)')
        axes[1, 0].set_title('尺度-位置图')
        
        # 杠杆 vs 残差平方（Cook阈值）
        axes[1, 1].scatter(leverage, stud_resid**2, c=cooks, cmap='viridis', alpha=0.7)
        n = len(data_expanded)
        cook_thresh = 4 / n if n > 0 else np.nan
        if pd.notna(cook_thresh):
            axes[1, 1].axhline(cook_thresh, color='red', linestyle='--', linewidth=1, label=f"Cook阈值={cook_thresh:.3f}")
        axes[1, 1].set_xlabel('杠杆值')
        axes[1, 1].set_ylabel('学生化残差平方')
        axes[1, 1].set_title('杠杆-残差图（颜色=Cook距离）')
        axes[1, 1].legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/diagnostic_plots.png', dpi=200)
        plt.close()
    except Exception as e:
        print(f"诊断图生成失败: {e}")
    
    # 3. 模型比较表
    print("\n3. 模型比较")
    print("=" * 50)
    
    comparison_results = {
        '模型': ['基础线性模型', '最优模型'],
        'R²': [f"{model_basic.rsquared:.4f}", f"{model_final.rsquared:.4f}"],
        '调整R²': [f"{model_basic.rsquared_adj:.4f}", f"{model_final.rsquared_adj:.4f}"],
        'AIC': [f"{model_basic.aic:.2f}", f"{model_final.aic:.2f}"],
        '变量数': [3, 3 + len(selected_vars)]
    }
    
    comparison_df = pd.DataFrame(comparison_results)
    print(comparison_df.to_string(index=False))
    
    # 保存结果
    with open(f'{output_dir}/modeling_results.txt', 'w', encoding='utf-8') as f:
        f.write("Y染色体浓度关系建模结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"数据筛选: 孕周10-25周\n")
        f.write(f"有效样本数: {len(data)}\n\n")
        
        f.write("1. 基础线性回归模型 (W + B + A)\n")
        f.write(f"   R²: {model_basic.rsquared:.4f}\n")
        f.write(f"   调整R²: {model_basic.rsquared_adj:.4f}\n")
        f.write(f"   AIC: {model_basic.aic:.2f}\n")
        f.write(f"   模型p值: {model_basic.f_pvalue:.6f}\n\n")
        
        if selected_vars:
            f.write("2. 逐步回归最优模型\n")
            f.write(f"   基础变量: {', '.join(base_vars)}\n")
            f.write(f"   新增项: {', '.join(selected_vars)}\n")
            f.write(f"   R²: {model_final.rsquared:.4f}\n")
            f.write(f"   调整R²: {model_final.rsquared_adj:.4f}\n")
            f.write(f"   AIC: {model_final.aic:.2f}\n")
            f.write("\n   逐步回归过程（每步新增项）：\n")
            if selection_trace:
                for step in selection_trace:
                    f.write(
                        f"     步骤{step['步骤']}: +{step['新增变量']} | AIC={step['AIC']:.4f}, R²={step['R²']:.6f}, 调整R²={step['调整R²']:.6f}, p={step['新增变量p值']:.6f}\n"
                    )
        else:
            f.write("2. 逐步回归未找到显著改善变量\n")
        
        # VIF摘要
        f.write("\n3. 多重共线性（VIF）检查\n")
        f.write("   变量  |   VIF\n")
        for _, row in vif_df.iterrows():
            f.write(f"   {row['变量']: <6}|   {row['VIF']:.4f}\n")
        f.write("   注: VIF>5 可能存在共线性, VIF>10 较强共线性\n")
        
        # 诊断图说明
        f.write("\n4. 诊断图\n")
        f.write("   已保存: diagnostic_plots.png（残差-拟合、QQ、尺度-位置、杠杆/Cook）\n")

    comparison_df.to_csv(f'{output_dir}/model_comparison.csv', index=False, encoding='utf-8-sig')
    
    # 保存逐步选择轨迹和最终模型系数
    if selection_trace:
        pd.DataFrame(selection_trace).to_csv(f'{output_dir}/selection_trace.csv', index=False, encoding='utf-8-sig')
    coef_df = pd.DataFrame({
        '变量': model_final.params.index,
        '系数': model_final.params.values,
        'p值': model_final.pvalues.values
    })
    coef_df.to_csv(f'{output_dir}/final_model_coefficients.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n结果已保存到 {output_dir}/ 目录")
    
    return model_basic, model_final if selected_vars else None

if __name__ == "__main__":
    regression_modeling()
