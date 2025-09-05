#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题二一键运行主入口（简洁版）
Pipeline:
  1) 数据处理 -> 生成 processed_data_problem2.csv
  2) BMI 分段 + 最佳时点（坐标下降优化） -> segment_optimized_summary.csv + segment_assignments.csv
  3) 可视化与报告 -> segment_plot.png
  4) （可选）鲁棒性分析 -> robustness_timing_summary.csv + robustness_boundaries_summary.csv
"""

import os
import sys
import argparse
import pandas as pd

# 让当前目录可直接 import 本目录下模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data_process import process_data as process_problem2_data  # type: ignore
from segmentation_optimization import run as optimize_segments  # type: ignore
from segmentation_report import run as report_segments  # type: ignore
from segmentation_robustness import run as robustness_analysis  # type: ignore


DEF_PROCESSED = os.path.join(BASE_DIR, 'processed_data_problem2.csv')
DEF_RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DEF_SUMMARY = os.path.join(DEF_RESULTS_DIR, 'segment_optimized_summary.csv')
DEF_ASSIGN = os.path.join(DEF_RESULTS_DIR, 'segment_assignments.csv')
DEF_PNG = os.path.join(DEF_RESULTS_DIR, 'segment_plot.png')
DEF_ROB_T = os.path.join(DEF_RESULTS_DIR, 'robustness_timing_summary.csv')
DEF_ROB_B = os.path.join(DEF_RESULTS_DIR, 'robustness_boundaries_summary.csv')


def run_pipeline(K: int = 5,
                 c: float = 0.90,
                 N_min: int = 20,
                 min_week: float = 12.0,
                 r1: float = 1.0,
                 r2: float = 3.0,
                 lam: float = 0.0,
                 delta: int = 3,
                 max_iters: int = 200,
                 tol: float = 1e-4,
                 force_reprocess: bool = False,
                 run_robustness: bool = False,
                 y_error_rate: float = 0.05,
                 bmi_error_rate: float = 0.00,
                 n_runs: int = 200):
    print('=' * 80)
    print('Step 1/3: 数据处理 -> 生成 processed_data_problem2.csv')
    print('-' * 80)
    if (not force_reprocess) and os.path.exists(DEF_PROCESSED):
        processed_df = pd.read_csv(DEF_PROCESSED)
        print(f"检测到已存在文件，跳过重处理：{DEF_PROCESSED}")
        print(f"样本数: {len(processed_df)} 名孕妇")
    else:
        processed_df = process_problem2_data()
        print(f"已生成: {DEF_PROCESSED} （{len(processed_df)} 名孕妇）")

    print('\n' + '=' * 80)
    print('Step 2/3: 分段与时点（坐标下降优化）')
    print('-' * 80)
    optimize_segments(
        input_csv=DEF_PROCESSED,
        output_csv=DEF_SUMMARY,
        K=K,
        c=c,
        N_min=N_min,
        min_week=min_week,
        r1=r1,
        r2=r2,
        lam=lam,
        delta=delta,
        max_iters=max_iters,
        tol=tol,
        assignments_out=DEF_ASSIGN,
    )

    print('\n' + '=' * 80)
    print('Step 3/3: 可视化与报告')
    print('-' * 80)
    report_segments(
        processed_csv=DEF_PROCESSED,
        summary_csv=DEF_SUMMARY,
        assignments_csv=DEF_ASSIGN,
        output_png=DEF_PNG,
    )

    if run_robustness:
        print('\n' + '=' * 80)
        print('附加 Step 4: 鲁棒性分析（Monte Carlo）')
        print('-' * 80)
        robustness_analysis(
            processed_csv=DEF_PROCESSED,
            K=K,
            c=c,
            N_min=N_min,
            min_week=min_week,
            r1=r1,
            r2=r2,
            lam=lam,
            delta=delta,
            max_iters=max_iters,
            tol=tol,
            y_error_rate=y_error_rate,
            bmi_error_rate=bmi_error_rate,
            n_runs=n_runs,
            out_timing_csv=DEF_ROB_T,
            out_bounds_csv=DEF_ROB_B,
        )

    print('\n' + '=' * 80)
    print('Pipeline 完成：')
    print(f'- 数据:           {DEF_PROCESSED}')
    print(f'- 分段摘要:       {DEF_SUMMARY}')
    print(f'- 分组明细:       {DEF_ASSIGN}')
    print(f'- 可视化:         {DEF_PNG}')
    if run_robustness:
        print(f'- 鲁棒性(T*):     {DEF_ROB_T}')
        print(f'- 鲁棒性(边界):   {DEF_ROB_B}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='问题二一键运行主入口（简洁版）')
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--c', type=float, default=0.90)
    parser.add_argument('--N_min', type=int, default=20)
    parser.add_argument('--min_week', type=float, default=12.0)
    parser.add_argument('--r1', type=float, default=1.0)
    parser.add_argument('--r2', type=float, default=3.0)
    parser.add_argument('--lam', type=float, default=0.0)
    parser.add_argument('--delta', type=int, default=3)
    parser.add_argument('--max_iters', type=int, default=200)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--force_reprocess', action='store_true')
    parser.add_argument('--run_robustness', action='store_true')
    parser.add_argument('--y_error_rate', type=float, default=0.05)
    parser.add_argument('--bmi_error_rate', type=float, default=0.00)
    parser.add_argument('--n_runs', type=int, default=200)
    args = parser.parse_args()

    run_pipeline(
        K=args.K,
        c=args.c,
        N_min=args.N_min,
        min_week=args.min_week,
        r1=args.r1,
        r2=args.r2,
        lam=args.lam,
        delta=args.delta,
        max_iters=args.max_iters,
        tol=args.tol,
        force_reprocess=args.force_reprocess,
        run_robustness=args.run_robustness,
        y_error_rate=args.y_error_rate,
        bmi_error_rate=args.bmi_error_rate,
        n_runs=args.n_runs,
    )
