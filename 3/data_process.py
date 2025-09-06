#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改版数据处理代码 - 保留身高和体重，进行标准化处理
"""

import pandas as pd
import numpy as np
from scipy.special import logit

def process_data():
    """按照data_process.md要求处理数据，同时保留身高和体重"""
    
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
    
    # 核心自变量（包括身高和体重）
    standardize_cols = {
        '孕周_小数': '孕周_标准化',
        df.columns[10]: 'BMI_标准化',  # K列 - 孕妇BMI
        df.columns[2]: '年龄_标准化',   # C列 - 年龄
        df.columns[3]: '身高_标准化',   # D列 - 身高
        df.columns[4]: '体重_标准化',   # E列 - 体重
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
            print(f"  {orig_col} -> {std_col}: 完成标准化 (均值={mean_val:.2f}, 标准差={std_val:.2f})")
    
    # 重新标准化所有染色体Z值
    z_cols = [df.columns[16], df.columns[17], df.columns[18], df.columns[19], df.columns[20]]  # Q,R,S,T,U列
    z_names = ['13号染色体Z值', '18号染色体Z值', '21号染色体Z值', 'X染色体Z值', 'Y染色体Z值']
    
    for i, col in enumerate(z_cols):
        if col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce')
            mean_val = data.mean()
            std_val = data.std()
            df[f'{z_names[i]}_重新标准化'] = (data - mean_val) / std_val
            print(f"  {z_names[i]} 重新标准化完成 (均值={mean_val:.3f}, 标准差={std_val:.3f})")
    
    print("数据标准化完成")
    
    # 2.4 对染色体浓度进行适当的比例数据处理
    print("2.4 对染色体浓度进行比例数据处理...")
    
    # 处理Y染色体浓度
    y_conc_col = df.columns[21]  # V列 - Y染色体浓度
    y_conc_data = pd.to_numeric(df[y_conc_col], errors='coerce')
    
    # 检查数据范围
    y_min, y_max = y_conc_data.min(), y_conc_data.max()
    print(f"  Y染色体浓度原始范围: [{y_min:.6f}, {y_max:.6f}]")
    
    # 数据已经是比例形式，只需进行数值裁剪以避免边界问题
    # 确保数据在(0,1)区间内，用于后续的分数logit建模
    epsilon = 1e-6
    y_processed = np.clip(y_conc_data, epsilon, 1-epsilon)
    df[y_conc_col] = y_processed
    print(f"  Y染色体浓度处理完成，处理后范围: [{df[y_conc_col].min():.6f}, {df[y_conc_col].max():.6f}]")
    
    # 处理X染色体浓度
    x_conc_col = df.columns[22]  # W列 - X染色体浓度  
    x_conc_data = pd.to_numeric(df[x_conc_col], errors='coerce')
    
    # 检查数据范围
    x_min, x_max = x_conc_data.min(), x_conc_data.max()
    print(f"  X染色体浓度原始范围: [{x_min:.6f}, {x_max:.6f}]")
    
    # 同样进行边界裁剪处理
    x_processed = np.clip(x_conc_data, epsilon, 1-epsilon)
    df[x_conc_col] = x_processed
    print(f"  X染色体浓度处理完成，处理后范围: [{df[x_conc_col].min():.6f}, {df[x_conc_col].max():.6f}]")
    
    print(f"  保持了原始比例含义，用于性别检测的4%阈值 = {0.04}")
    
   
    print("\n3. 保留有价值的数据列...")
    
    # 保留的列 - 避免重复，只保留标准化后的值
    keep_columns = [
        # 基本标识信息
        df.columns[1],  # 孕妇代码 (用于标识)
        
        # 不需要标准化的基本信息
        df.columns[6],  # IVF妊娠（分类变量，不需标准化）
        df.columns[8],  # 检测抽血次数（计数变量）
        
        # 标准化后的核心变量（不保留原始值）
        '孕周_标准化',     # 核心自变量
        'BMI_标准化',     # 核心自变量  
        '年龄_标准化',     # 核心自变量
        '身高_标准化',     # 身高标准化
        '体重_标准化',     # 体重标准化
        '怀孕次数_标准化', # 其他自变量
        '生产次数_标准化', # 其他自变量
        
        # 数据质量控制变量（只保留标准化值）
        '唯一比对读段比例_标准化', # 计算得出并标准化的比例
        '总读段数_标准化', 
        '比对比例_标准化', 
        '重复读段比例_标准化', 
        'GC含量_标准化',   
        '过滤读段比例_标准化', 
        
        # 染色体Z值（只保留重新标准化的版本）
        '13号染色体Z值_重新标准化',
        '18号染色体Z值_重新标准化',
        '21号染色体Z值_重新标准化',
        'X染色体Z值_重新标准化',
        'Y染色体Z值_重新标准化',
        
        # 染色体浓度信息（比例数据，保留处理后的版本）
        df.columns[21], # Y染色体浓度 (比例数据，已进行边界裁剪处理)
        df.columns[22], # X染色体浓度（也可能有用）
        
        # 各染色体GC含量（原始比例值，有意义保留）
        df.columns[23], # 13号染色体的GC含量
        df.columns[24], # 18号染色体的GC含量
        df.columns[25], # 21号染色体的GC含量
        
        # 临床相关信息
        df.columns[27], # 染色体的非整倍体（重要的临床信息）
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
    
    print(f"删除了 {removed_cols} 个列，保留 {len(df.columns)} 列（避免重复，只保留必要信息）")
    print("保留的列分类:")
    
    # 按类别显示保留的列
    basic_info_cols = [col for col in df.columns if any(x in str(col) for x in ['代码', 'IVF', '抽血']) and '_标准化' not in col]
    standardized_cols = [col for col in df.columns if '_标准化' in col and '_重新标准化' not in col]
    restandard_z_cols = [col for col in df.columns if '_重新标准化' in col]
    concentration_cols = [col for col in df.columns if '浓度' in str(col)]
    gc_content_cols = [col for col in df.columns if 'GC含量' in str(col) and '_标准化' not in col]
    other_cols = [col for col in df.columns if col not in basic_info_cols + standardized_cols + restandard_z_cols + concentration_cols + gc_content_cols]
    
    print(f"  基本信息 ({len(basic_info_cols)}列): {', '.join(basic_info_cols)}")
    print(f"  标准化变量 ({len(standardized_cols)}列): {', '.join(standardized_cols[:3])}{'...' if len(standardized_cols) > 3 else ''}")
    print(f"  重新标准化Z值 ({len(restandard_z_cols)}列): {', '.join(restandard_z_cols[:3])}{'...' if len(restandard_z_cols) > 3 else ''}")
    print(f"  染色体浓度 ({len(concentration_cols)}列): {', '.join(concentration_cols)}")
    print(f"  GC含量 ({len(gc_content_cols)}列): {', '.join(gc_content_cols)}")
    if other_cols:
        print(f"  其他 ({len(other_cols)}列): {', '.join(other_cols)}")
    
    # 保存处理后的数据到3文件夹
    print("\n保存处理后的数据...")
    output_file = '/Users/Mac/Downloads/mm/3/processed_data.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"数据已保存至: {output_file}")
    
    # 输出处理摘要
    print("\n" + "="*50)
    print("数据处理摘要")
    print("="*50)
    print(f"最终数据量: {len(df)}")
    print(f"数据列数: {len(df.columns)}")
    print("\n新增的处理变量:")
    print("- 孕周_小数")
    print("- 唯一比对读段比例")
    print("- 身高_标准化")
    print("- 体重_标准化")
    print("- 怀孕次数_清洗")
    print("- 生产次数_清洗")
    print("\n变换的变量:")
    print("- Y染色体浓度 (已进行边界裁剪处理，保持原始比例含义)")
    
    standardized_vars = [col for col in df.columns if '_标准化' in col or '_重新标准化' in col]
    print(f"\n标准化变量数量: {len(standardized_vars)}")
    print("主要标准化变量:")
    for var in standardized_vars[:10]:  # 只显示前10个
        print(f"- {var}")
    if len(standardized_vars) > 10:
        print(f"... 还有{len(standardized_vars) - 10}个标准化变量")
    
    # 显示数据保留策略
    print(f"\n数据保留策略:")
    print(f"- 避免重复：对于需要标准化的变量，只保留标准化后的值")
    print(f"- 保留关键染色体信息（重新标准化的Z值、浓度、GC含量）")
    print(f"- 保留数据质量控制变量的标准化版本")
    print(f"- 保留重要临床信息（IVF妊娠、非整倍体等）")
    print(f"- 删除了原始值以避免冗余，节省存储空间")
    
    return df

if __name__ == "__main__":
    process_data()
