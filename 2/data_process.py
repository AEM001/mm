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
    
    # 1.1 剔除异常生理样本 - 剔除标记为"否"的样本
    print("1.1 剔除异常生理样本...")
    health_col = '胎儿是否健康'
    before_health = len(df)
    df = df[df[health_col] == '是'].copy()
    print(f"剔除不健康胎儿样本: {before_health - len(df)} 个，剩余: {len(df)}")
    
    # 1.2 剔除低质量检测数据 - 使用3σ原则
    print("1.2 剔除低质量检测数据...")
    
    # 使用正确的列名
    l_col = '原始读段数'
    m_col = '在参考基因组上比对的比例'
    n_col = '重复读段的比例'
    o_col = '唯一比对的读段数  '  # 注意：这个列名后面有空格
    p_col = 'GC含量'
    aa_col = '被过滤掉读段数的比例'
    
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
    j_col = '检测孕周'
    
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
    
    # 3. 按照问题二要求进行数据筛选
    print("\n3. 问题二数据筛选")
    
    # 3.1 仅考虑Y染色体浓度达到4%的对象
    print("3.1 筛选Y染色体浓度>=4%的样本...")
    before_y_filter = len(df)
    
    # 使用正确的列名获取Y染色体浓度
    y_concentration_col = 'Y染色体浓度'
    df['Y染色体浓度_数值'] = pd.to_numeric(df[y_concentration_col], errors='coerce')
    
    # 筛选Y染色体浓度>=4% (0.04)的样本
    df = df[df['Y染色体浓度_数值'] >= 0.04].copy()
    print(f"筛选Y染色体浓度>=4%: 剔除 {before_y_filter - len(df)} 个样本，剩余: {len(df)}")
    
    # 3.2 筛选孕周为10-25的部分数据
    print("3.2 筛选孕周10-25周的样本...")
    before_week_filter = len(df)
    
    # 筛选孕周在10-25周范围内的样本
    df = df[(df['孕周_小数'] >= 10) & (df['孕周_小数'] <= 25)].copy()
    print(f"筛选孕周10-25周: 剔除 {before_week_filter - len(df)} 个样本，剩余: {len(df)}")
    
    # 4. 保留指定数据列
    print("\n4. 提取最终数据")
    
    # 4.1 获取需要保留的列
    b_col = '孕妇代码'
    k_col = '孕妇BMI'
    
    # 创建最终数据集，包含：孕妇代码、Y染色体浓度、孕周、BMI
    final_data = pd.DataFrame({
        '孕妇代码': df[b_col],
        'Y染色体浓度': df['Y染色体浓度_数值'],
        '孕周': df['孕周_小数'],
        'BMI': pd.to_numeric(df[k_col], errors='coerce')
    })
    
    # 移除任何包含缺失值的行
    before_na_drop = len(final_data)
    final_data = final_data.dropna()
    print(f"移除缺失值: 剔除 {before_na_drop - len(final_data)} 个样本，最终数据量: {len(final_data)}")
    
    # 5. 数据统计和保存
    print("\n5. 数据统计")
    print("最终数据集统计信息:")
    print(final_data.describe())
    
    print(f"\n各变量范围:")
    print(f"孕妇代码数量: {final_data['孕妇代码'].nunique()} 个唯一代码")
    print(f"Y染色体浓度: {final_data['Y染色体浓度'].min():.4f} - {final_data['Y染色体浓度'].max():.4f}")
    print(f"孕周: {final_data['孕周'].min():.1f} - {final_data['孕周'].max():.1f}")
    print(f"BMI: {final_data['BMI'].min():.2f} - {final_data['BMI'].max():.2f}")
    
    print(f"\n前5个样本预览:")
    print(final_data.head())
    
    # 保存处理后的数据
    output_file = '/Users/Mac/Downloads/mm/2/processed_data_problem2.csv'
    final_data.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n数据已保存至: {output_file}")
    
    return final_data

if __name__ == "__main__":
    processed_data = process_data()
