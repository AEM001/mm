#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分组报告与可视化（简洁版）
- 输入：
  - processed CSV: /Users/Mac/Downloads/mm/2/processed_data_problem2.csv （包含 BMI, 最早达标孕周）
  - summary CSV:   /Users/Mac/Downloads/mm/2/results/segment_optimized_summary.csv
  - assignments:   /Users/Mac/Downloads/mm/2/results/segment_assignments.csv （可选，优先使用）
- 输出：
  - /Users/Mac/Downloads/mm/2/results/segment_plot.png
"""

import os
import re
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 项目路径与中文字体
sys.path.append('/Users/Mac/Downloads/mm')
from set_chinese_font import set_chinese_font
set_chinese_font()

def parse_interval(interval_str: str):
    """解析形如 "[20.7, 31.5]" 或 "[20.7, 31.5)" 的区间字符串，返回 (left, right, right_inclusive)
    若解析失败，返回 (nan, nan, True)
    """
    try:
        s = interval_str.strip()
        right_inclusive = s.endswith(']')
        nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", s)
        if len(nums) >= 2:
            left = float(nums[0]); right = float(nums[1])
            return left, right, right_inclusive
    except Exception:
        pass
    return np.nan, np.nan, True


def assign_groups_from_summary(df, summary_df):
    """当没有 assignments 明细时，基于 summary 的 BMI 区间为样本分配组别。
    规则：所有中间组采用左闭右开；最后一组右闭。
    返回包含列：BMI, 最早达标孕周, group, group_bmi_min, group_bmi_max, T_star
    """
    # 读取每组的区间与 T*（按组号排序）
    groups = []
    for _, row in summary_df.sort_values('组号').iterrows():
        interval_str = str(row['BMI区间'])
        left, right, right_incl = parse_interval(interval_str)
        # 强制约定：中间组右开，最后一组右闭
        groups.append({
            'group': int(row['组号']),
            'left': float(left),
            'right': float(right),
            'T_star': float(row['T_star(周)'])
        })
    # 分配
    out_rows = []
    max_group = max(g['group'] for g in groups)
    for _, r in df.iterrows():
        b = float(r['BMI']); w = float(r['最早达标孕周'])
        g_assigned = None
        g_left = np.nan; g_right = np.nan; g_t = np.nan
        for g in groups:
            if g['group'] < max_group:
                if (b >= g['left']) and (b < g['right']):
                    g_assigned = g['group']; g_left = g['left']; g_right = g['right']; g_t = g['T_star']
                    break
            else:
                if (b >= g['left']) and (b <= g['right']):
                    g_assigned = g['group']; g_left = g['left']; g_right = g['right']; g_t = g['T_star']
                    break
        out_rows.append({
            'BMI': b,
            '最早达标孕周': w,
            'group': g_assigned,
            'group_bmi_min': g_left,
            'group_bmi_max': g_right,
            'T_star': g_t,
        })
    return pd.DataFrame(out_rows)


def plot_segments(assign_df: pd.DataFrame, output_png: str, title: str = 'BMI分段与最佳时点'):
    plt.figure(figsize=(10, 5))
    # 绘制散点
    groups = sorted(assign_df['group'].dropna().unique())
    colors = plt.cm.Set1(np.linspace(0, 1, max(3, len(groups))))
    for i, g in enumerate(groups):
        sub = assign_df[assign_df['group'] == g]
        plt.scatter(sub['BMI'], sub['最早达标孕周'], s=20, alpha=0.7, color=colors[i], label=f'第{int(g)}组')
        # 水平线 T*
        if len(sub) > 0:
            x_min = float(sub['group_bmi_min'].min())
            x_max = float(sub['group_bmi_max'].max())
            t = float(sub['T_star'].iloc[0])
            plt.hlines(y=t, xmin=x_min, xmax=x_max, colors=colors[i], linestyles='--')
    # 竖线：组边界
    bounds = sorted(assign_df.groupby('group')['group_bmi_max'].max().values[:-1])
    for b in bounds:
        plt.axvline(x=b, color='k', linestyle=':', alpha=0.6)
    plt.xlabel('BMI')
    plt.ylabel('最早达标孕周')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if len(groups) <= 12:
        plt.legend()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'可视化已保存到: {output_png}')


def run(processed_csv: str,
        summary_csv: str,
        assignments_csv: str,
        output_png: str):
    df = pd.read_csv(processed_csv)
    summary_df = pd.read_csv(summary_csv)
    # 优先使用 assignments
    assign_df = None
    try:
        if assignments_csv and os.path.exists(assignments_csv):
            assign_df = pd.read_csv(assignments_csv)
    except Exception:
        assign_df = None
    if assign_df is None or assign_df.empty:
        print('未找到有效的 assignments 文件，改用 summary 的区间生成分配...')
        df_small = df[['BMI', '最早达标孕周']].dropna().copy()
        assign_df = assign_groups_from_summary(df_small, summary_df)
    # 绘图
    title = 'BMI分段与最佳时点（坐标下降优化）'
    plot_segments(assign_df, output_png=output_png, title=title)

    # 输出汇总
    grp = assign_df.groupby('group').agg(
        样本数=('BMI', 'size'),
        BMI均值=('BMI', 'mean'),
        区间左=('group_bmi_min', 'min'),
        区间右=('group_bmi_max', 'max'),
        最佳时点_T=('T_star', 'first'),
    ).reset_index().sort_values('group')
    print('\n分组汇总:')
    print(grp.to_string(index=False, float_format='%.3f'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分组报告与可视化（简洁版）')
    parser.add_argument('--processed', type=str, default='/Users/Mac/Downloads/mm/2/processed_data_problem2.csv')
    parser.add_argument('--summary', type=str, default='/Users/Mac/Downloads/mm/2/results/segment_optimized_summary.csv')
    parser.add_argument('--assignments', type=str, default='/Users/Mac/Downloads/mm/2/results/segment_assignments.csv')
    parser.add_argument('--output_png', type=str, default='/Users/Mac/Downloads/mm/2/results/segment_plot.png')
    args = parser.parse_args()
    run(args.processed, args.summary, args.assignments, args.output_png)
