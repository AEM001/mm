import pandas as pd
import numpy as np
from scipy.special import logit

def process_data():
    df = pd.read_excel('../附件.xlsx', sheet_name='男胎检测数据')
    
    ae_col = df.columns[-1]
    df = df[df[ae_col] == '是'].copy()
    
    l_col = df.columns[11]
    m_col = df.columns[12]
    n_col = df.columns[13]
    o_col = df.columns[14]
    p_col = df.columns[15]
    aa_col = df.columns[26]
    
    low_threshold_cols = [l_col, m_col, o_col, p_col]
    for col in low_threshold_cols:
        col_data = pd.to_numeric(df[col], errors='coerce')
        mean_val = col_data.mean()
        std_val = col_data.std()
        threshold = mean_val - 3 * std_val
        df = df[col_data >= threshold].copy()
    
    high_threshold_cols = [n_col, aa_col]
    for col in high_threshold_cols:
        col_data = pd.to_numeric(df[col], errors='coerce')
        mean_val = col_data.mean()
        std_val = col_data.std()
        threshold = mean_val + 3 * std_val
        df = df[col_data <= threshold].copy()
    
    j_col = df.columns[9]
    
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
    
    df['孕周_小数'] = df[j_col].apply(convert_gestational_week)
    df['唯一比对读段比例'] = pd.to_numeric(df[o_col], errors='coerce') / pd.to_numeric(df[l_col], errors='coerce')
    
    if len(df.columns) > 28:
        pregnancy_col = df.columns[28]
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
        
        df['怀孕次数_清洗'] = df[pregnancy_col].apply(clean_pregnancy_count)
    
    if len(df.columns) > 29:
        birth_col = df.columns[29]
        df['生产次数_清洗'] = pd.to_numeric(df[birth_col], errors='coerce')
    
    standardize_cols = {
        '孕周_小数': '孕周_标准化',
        df.columns[10]: 'BMI_标准化',
        df.columns[2]: '年龄_标准化',
    }
    
    if '怀孕次数_清洗' in df.columns:
        standardize_cols['怀孕次数_清洗'] = '怀孕次数_标准化'
    if '生产次数_清洗' in df.columns:
        standardize_cols['生产次数_清洗'] = '生产次数_标准化'
    
    quality_cols = {
        l_col: '总读段数_标准化',
        m_col: '比对比例_标准化', 
        n_col: '重复读段比例_标准化',
        '唯一比对读段比例': '唯一比对读段比例_标准化',
        p_col: 'GC含量_标准化',
        aa_col: '过滤读段比例_标准化'
    }
    
    standardize_cols.update(quality_cols)
    
    for orig_col, std_col in standardize_cols.items():
        if orig_col in df.columns or orig_col == '孕周_小数' or orig_col == '唯一比对读段比例':
            data = pd.to_numeric(df[orig_col], errors='coerce')
            mean_val = data.mean()
            std_val = data.std()
            df[std_col] = (data - mean_val) / std_val
    
    z_cols = [df.columns[16], df.columns[17], df.columns[18], df.columns[19], df.columns[20]]
    z_names = ['13号染色体Z值', '18号染色体Z值', '21号染色体Z值', 'X染色体Z值', 'Y染色体Z值']
    
    for i, col in enumerate(z_cols):
        if col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce')
            mean_val = data.mean()
            std_val = data.std()
            df[f'{z_names[i]}_重新标准化'] = (data - mean_val) / std_val
    
    y_conc_col = df.columns[21]
    y_conc_data = pd.to_numeric(df[y_conc_col], errors='coerce')
    
    y_min, y_max = y_conc_data.min(), y_conc_data.max()
    
    epsilon = 1e-6
    y_processed = np.clip(y_conc_data, epsilon, 1-epsilon)
    
    df[y_conc_col] = y_processed
    
    keep_columns = [
        df.columns[1],
        df.columns[21],
        '孕周_标准化',
        'BMI_标准化',
        '年龄_标准化',
        '怀孕次数_标准化',
        '生产次数_标准化',
        '总读段数_标准化',
        '比对比例_标准化',
        '重复读段比例_标准化',
        '唯一比对读段比例_标准化',
        'GC含量_标准化',
        '过滤读段比例_标准化',
        'Y染色体Z值_重新标准化'
    ]
    
    existing_keep_columns = []
    for col in keep_columns:
        if col in df.columns:
            existing_keep_columns.append(col)
    
    df = df[existing_keep_columns].copy()
    df.to_csv('processed_data.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    process_data()
