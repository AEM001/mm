#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格按照data_process.md要求进行数据处理
"""

import pandas as pd
import numpy as np
from scipy.special import logit

def process_data():
    """按照data_process.md要求处理数据"""
    
    # 读取男胎检测数据
    print("读取数据...")
    df = pd.read_excel('/Users/Mac/Downloads/mm/附件.xlsx', sheet_name='男胎检测数据')
    print(f"原始数据量: {len(df)}")
    
    # 1. 数据清洗
    print("\n1. 数据清洗")
    
    # 1.1 剔除异常生理样本 - 剔除AE列标记为"否"的样本
    print("1.1 剔除异常生理样本...")
    ae_col = df.columns[-1]  # AE列是最后一列 (胎儿是否健康)
    before_health = len(df)
    df = df[df[ae_col] == '是'].copy()
    print(f"剔除不健康胎儿样本: {before_health - len(df)} 个，剩余: {len(df)}")
    
    # 1.2 剔除低质量检测数据 - 使用3σ原则
    print("1.2 剔除低质量检测数据...")
    
    # 找到对应的列索引
    l_col = df.columns[11]  # L列 - 原始读段数
    m_col = df.columns[12]  # M列 - 在参考基因组上比对的比例  
    n_col = df.columns[13]  # N列 - 重复读段的比例
    o_col = df.columns[14]  # O列 - 唯一比对的读段数
    p_col = df.columns[15]  # P列 - GC含量
    aa_col = df.columns[26] # AA列 - 被过滤掉读段数的比例
    
    # 剔除低于 μ - 3σ 的样本 (L, M, O, P列)
    low_threshold_cols = [l_col, m_col, o_col, p_col]
    for col in low_threshold_cols:
        before_count = len(df)
        col_data = pd.to_numeric(df[col], errors='coerce')
        mean_val = col_data.mean()
        std_val = col_data.std()
        threshold = mean_val - 3 * std_val
        df = df[col_data >= threshold].copy()
        print(f"  {col}: 剔除 {before_count - len(df)} 个低于阈值({threshold:.4f})的样本")
    
    # 剔除高于 μ + 3σ 的样本 (N, AA列)
    high_threshold_cols = [n_col, aa_col]
    for col in high_threshold_cols:
        before_count = len(df)
        col_data = pd.to_numeric(df[col], errors='coerce')
        mean_val = col_data.mean()
        std_val = col_data.std()
        threshold = mean_val + 3 * std_val
        df = df[col_data <= threshold].copy()
        print(f"  {col}: 剔除 {before_count - len(df)} 个高于阈值({threshold:.4f})的样本")
    
    print(f"数据清洗完成，最终数据量: {len(df)}")
    
    # 2. 数据预处理
    print("\n2. 数据预处理")
    
    # 2.1 变量格式统一 - 转换孕周格式
    print("2.1 变量格式统一...")
    j_col = df.columns[9]  # J列 - 检测孕周
    
    def convert_gestational_week(week_str):
        """将孕周格式 '11w+6' 转换为小数周数"""
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
    
    df['孕周_小数'] = df[j_col].apply(convert_gestational_week)
    print(f"孕周格式转换完成，有效数据: {df['孕周_小数'].notna().sum()} 个")
    
    # 2.2 特征转换 - O列/L列 = 唯一比对读段比例
    print("2.2 特征转换...")
    df['唯一比对读段比例'] = pd.to_numeric(df[o_col], errors='coerce') / pd.to_numeric(df[l_col], errors='coerce')
    print("唯一比对读段比例计算完成")
    
    # 2.3 数据标准化
    print("2.3 数据标准化...")
    
    # 处理怀孕次数和生产次数的特殊值
    print("  处理怀孕次数和生产次数的特殊值...")
    
    # 处理怀孕次数列 (AC列) - 将">=3"等转换为数值3
    if len(df.columns) > 28:
        pregnancy_col = df.columns[28]  # AC列
        def clean_pregnancy_count(value):
            if pd.isna(value):
                return np.nan
            value_str = str(value).strip()
            # 处理 ≥3 或 >=3 的情况
            if '≥' in value_str or '>=' in value_str:
                # 提取数字部分
                import re
                numbers = re.findall(r'\d+', value_str)
                if numbers:
                    return float(numbers[0])
            try:
                return float(value_str)
            except:
                return np.nan
        
        df['怀孕次数_清洗'] = df[pregnancy_col].apply(clean_pregnancy_count)
        print(f"    怀孕次数清洗完成，转换了包含'>='等特殊值的数据")
    
    # 处理生产次数列 (AD列)  
    if len(df.columns) > 29:
        birth_col = df.columns[29]  # AD列
        df['生产次数_清洗'] = pd.to_numeric(df[birth_col], errors='coerce')
        print(f"    生产次数清洗完成")
    
    # 核心自变量
    standardize_cols = {
        '孕周_小数': '孕周_标准化',
        df.columns[10]: 'BMI_标准化',  # K列 - 孕妇BMI
        df.columns[2]: '年龄_标准化',   # C列 - 年龄
    }
    
    # 其他可探索的自变量 - 使用清洗后的数据
    if '怀孕次数_清洗' in df.columns:
        standardize_cols['怀孕次数_清洗'] = '怀孕次数_标准化'
    if '生产次数_清洗' in df.columns:
        standardize_cols['生产次数_清洗'] = '生产次数_标准化'
    
    # 数据质量控制变量
    quality_cols = {
        l_col: '总读段数_标准化',
        m_col: '比对比例_标准化', 
        n_col: '重复读段比例_标准化',
        '唯一比对读段比例': '唯一比对读段比例_标准化',
        p_col: 'GC含量_标准化',
        aa_col: '过滤读段比例_标准化'
    }
    
    standardize_cols.update(quality_cols)
    
    # 执行Z-score标准化
    for orig_col, std_col in standardize_cols.items():
        if orig_col in df.columns or orig_col == '孕周_小数' or orig_col == '唯一比对读段比例':
            data = pd.to_numeric(df[orig_col], errors='coerce')
            mean_val = data.mean()
            std_val = data.std()
            df[std_col] = (data - mean_val) / std_val
            print(f"  {orig_col} -> {std_col}: 完成标准化")
    
    # 重新标准化染色体Z值
    z_cols = [df.columns[16], df.columns[17], df.columns[18], df.columns[19], df.columns[20]]  # Q,R,S,T,U列
    z_names = ['13号染色体Z值', '18号染色体Z值', '21号染色体Z值', 'X染色体Z值', 'Y染色体Z值']
    
    for i, col in enumerate(z_cols):
        if col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce')
            mean_val = data.mean()
            std_val = data.std()
            df[f'{z_names[i]}_重新标准化'] = (data - mean_val) / std_val
            print(f"  {z_names[i]} 重新标准化完成")
    
    print("数据标准化完成")
    
    # 2.4 对Y染色体浓度进行logit变换
    print("2.4 对Y染色体浓度进行logit变换...")
    y_conc_col = df.columns[21]  # V列 - Y染色体浓度
    y_conc_data = pd.to_numeric(df[y_conc_col], errors='coerce')
    
    # 确保数据在(0,1)区间内，如果不是则进行调整
    # 先检查数据范围
    y_min, y_max = y_conc_data.min(), y_conc_data.max()
    print(f"  Y染色体浓度原始范围: [{y_min:.6f}, {y_max:.6f}]")
    
    # 如果数据不在(0,1)区间，进行归一化
    if y_min <= 0 or y_max >= 1:
        # 使用min-max归一化到(epsilon, 1-epsilon)区间，避免logit函数的边界问题
        epsilon = 1e-6
        y_normalized = (y_conc_data - y_min) / (y_max - y_min)
        y_normalized = y_normalized * (1 - 2*epsilon) + epsilon
        print(f"  数据已归一化到({epsilon}, {1-epsilon})区间")
    else:
        y_normalized = y_conc_data
        # 处理边界值，避免logit函数的数值问题
        epsilon = 1e-6
        y_normalized = np.clip(y_normalized, epsilon, 1-epsilon)
    
    # 应用logit变换，直接覆盖原列以保持名称一致
    df[y_conc_col] = logit(y_normalized)
    print(f"  logit变换完成，变换后范围: [{df[y_conc_col].min():.6f}, {df[y_conc_col].max():.6f}]")
    
    # 3. 删除冗余和无关的数据列
    print("\n3. 删除冗余和无关的数据列...")
    
    # 保留的列 - 只保留与问题一相关的核心变量
    keep_columns = [
        df.columns[1],  # 孕妇代码 (用于标识)
        df.columns[21], # Y染色体浓度 (经过logit变换的目标变量V列)
        '孕周_标准化',     # 核心自变量
        'BMI_标准化',     # 核心自变量  
        '年龄_标准化',     # 核心自变量
        '怀孕次数_标准化', # 其他自变量
        '生产次数_标准化', # 其他自变量
        '总读段数_标准化', # 数据质量控制变量
        '比对比例_标准化', # 数据质量控制变量
        '重复读段比例_标准化', # 数据质量控制变量
        '唯一比对读段比例_标准化', # 数据质量控制变量
        'GC含量_标准化',   # 数据质量控制变量
        '过滤读段比例_标准化', # 数据质量控制变量
        'Y染色体Z值_重新标准化'  # Y染色体相关Z值
    ]
    
    # 检查列是否存在并过滤
    existing_keep_columns = []
    for col in keep_columns:
        if col in df.columns:
            existing_keep_columns.append(col)
        else:
            print(f"  警告: 列 '{col}' 不存在")
    
    # 只保留需要的列
    original_cols = len(df.columns)
    df = df[existing_keep_columns].copy()
    removed_cols = original_cols - len(df.columns)
    
    print(f"删除了 {removed_cols} 个冗余/无关列，保留 {len(df.columns)} 列")
    print("保留的列:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # 保存处理后的数据
    print("\n保存处理后的数据...")
    df.to_csv('/Users/Mac/Downloads/mm/1/processed_data.csv', index=False, encoding='utf-8-sig')
    print("数据已保存至: processed_data.csv")
    
    # 输出处理摘要
    print("\n" + "="*50)
    print("数据处理摘要")
    print("="*50)
    print(f"最终数据量: {len(df)}")
    print(f"数据列数: {len(df.columns)}")
    print("\n新增的处理变量:")
    print("- 孕周_小数")
    print("- 唯一比对读段比例")
    print("\n变换的变量:")
    print("- Y染色体浓度 (已进行logit变换，保持原列名)") 
    
    standardized_vars = [col for col in df.columns if '_标准化' in col or '_重新标准化' in col]
    print(f"\n标准化变量数量: {len(standardized_vars)}")
    for var in standardized_vars:
        print(f"- {var}")

if __name__ == "__main__":
    process_data()
