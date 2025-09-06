#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

def process_data():
    """按照问题二要求处理数据"""
    
    # 读取男胎检测数据
    print("读取数据...")
    df = pd.read_excel('/Users/Mac/Downloads/mm/附件.xlsx', sheet_name='男胎检测数据')
    print(f"原始数据量: {len(df)} 条记录")
    print(f"包含孕妇数量: {df['孕妇代码'].nunique()} 名")
    
    # 1. 基础数据清洗
    print("\n1. 基础数据清洗")
    
    # 1.1 剔除不健康胎儿样本
    print("1.1 剔除不健康胎儿样本...")
    before_health = len(df)
    df = df[df['胎儿是否健康'] == '是'].copy()
    print(f"剔除不健康胎儿样本: {before_health - len(df)} 条记录，剩余: {len(df)} 条")
    
    # 1.2 数据预处理 - 转换孕周格式
    print("1.2 转换孕周格式...")
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
    
    df['孕周_小数'] = df['检测孕周'].apply(convert_gestational_week)
    df['Y染色体浓度_数值'] = pd.to_numeric(df['Y染色体浓度'], errors='coerce')
    df['BMI_数值'] = pd.to_numeric(df['孕妇BMI'], errors='coerce')
    
    # 移除关键数据缺失的记录
    before_na = len(df)
    df = df.dropna(subset=['孕周_小数', 'Y染色体浓度_数值', 'BMI_数值'])
    print(f"移除关键数据缺失记录: {before_na - len(df)} 条，剩余: {len(df)} 条")
    
    # 2. 按孕妇分组处理
    print("\n2. 按孕妇分组处理")
    
    # 按孕妇代码分组
    grouped = df.groupby('孕妇代码')
    print(f"共有 {len(grouped)} 名孕妇")
    
    # 存储最终结果
    final_results = []
    excluded_women = []
    
    for woman_code, woman_data in grouped:
        # 按孕周排序（确保按时间顺序处理）
        woman_data = woman_data.sort_values('孕周_小数').reset_index(drop=True)
        
        # 检查是否有任何检测达到4%
        has_qualified = (woman_data['Y染色体浓度_数值'] >= 0.04).any()
        
        if not has_qualified:
            # 如果所有检测都未达到4%，剔除该孕妇
            excluded_women.append(woman_code)
            continue
        
        # 找到第一次Y染色体浓度>=4%的记录
        qualified_records = woman_data[woman_data['Y染色体浓度_数值'] >= 0.04]
        
        if len(qualified_records) == len(woman_data):
            # 特殊情况：如果所有检测都>=4%，取第一次检测数据
            first_record = woman_data.iloc[0]
        else:
            # 一般情况：取第一次达标的检测数据
            first_record = qualified_records.iloc[0]
        
        # 添加到结果中
        final_results.append({
            '孕妇代码': woman_code,
            'BMI': first_record['BMI_数值'],
            '最早达标孕周': first_record['孕周_小数'],
            'Y染色体浓度': first_record['Y染色体浓度_数值']
        })
    
    print(f"剔除从未达标的孕妇: {len(excluded_women)} 名")
    print(f"保留的孕妇: {len(final_results)} 名")
    
    # 3. 创建最终数据集
    print("\n3. 创建最终数据集")
    final_data = pd.DataFrame(final_results)
    
    # 筛选孕周在10-25周范围内的数据
    before_week_filter = len(final_data)
    final_data = final_data[(final_data['最早达标孕周'] >= 10) & (final_data['最早达标孕周'] <= 25)].copy()
    print(f"筛选孕周10-25周: 剔除 {before_week_filter - len(final_data)} 名孕妇，剩余: {len(final_data)} 名")
    
    # 4. 数据统计和保存
    print("\n4. 数据统计")
    print("最终数据集统计信息:")
    print(final_data.describe())
    
    print(f"\n各变量范围:")
    print(f"孕妇数量: {len(final_data)} 名")
    print(f"BMI: {final_data['BMI'].min():.2f} - {final_data['BMI'].max():.2f}")
    print(f"最早达标孕周: {final_data['最早达标孕周'].min():.1f} - {final_data['最早达标孕周'].max():.1f}")
    print(f"Y染色体浓度: {final_data['Y染色体浓度'].min():.4f} - {final_data['Y染色体浓度'].max():.4f}")
    
    print(f"\n前10个样本预览:")
    print(final_data.head(10))
    
    # 保存处理后的数据
    output_file = '/Users/Mac/Downloads/mm/2/processed_data_problem2.csv'
    final_data.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n数据已保存至: {output_file}")
    
    # 输出一些被剔除的孕妇示例
    if excluded_women:
        print(f"\n被剔除的孕妇示例（前5名）: {excluded_women[:5]}")
    
    return final_data

if __name__ == "__main__":
    processed_data = process_data()
