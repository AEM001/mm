#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMI 分段与最佳时点的鲁棒性分析（简洁版）
- 对 BMI 与 最早达标孕周 注入随机误差，重复运行坐标下降优化，统计 T* 与边界的稳定性。
- 依赖 segmentation_optimization.py
输出：
  - /Users/Mac/Downloads/mm/2/results/robustness_timing_summary.csv
  - /Users/Mac/Downloads/mm/2/results/robustness_boundaries_summary.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import sys

# 便于导入同目录脚本
sys.path.append('/Users/Mac/Downloads/mm/2')
from segmentation_optimization import Params, coordinate_descent  # type: ignore


def add_random_error(arr: np.ndarray, rate: float) -> np.ndarray:
    if rate is None or rate <= 0:
        return arr.copy()
    noise = np.random.uniform(-1.0, 1.0, size=arr.shape)
    return arr * (1.0 + rate * noise)


def run(processed_csv: str,
        K: int = 5,
        c: float = 0.90,
        N_min: int = 20,
        min_week: float = 12.0,
        r1: float = 1.0,
        r2: float = 3.0,
        lam: float = 0.0,
        delta: int = 3,
        max_iters: int = 200,
        tol: float = 1e-4,
        y_error_rate: float = 0.05,
        bmi_error_rate: float = 0.00,
        n_runs: int = 200,
        out_timing_csv: str = '/Users/Mac/Downloads/mm/2/results/robustness_timing_summary.csv',
        out_bounds_csv: str = '/Users/Mac/Downloads/mm/2/results/robustness_boundaries_summary.csv'):
    # 加载与清理
    df = pd.read_csv(processed_csv)
    df = df[['BMI', '最早达标孕周']].dropna().copy()

    # 基线 BMI 范围用于裁剪
    bmi_min0, bmi_max0 = float(df['BMI'].min()), float(df['BMI'].max())

    params = Params(c=c, min_week=min_week, r1=r1, r2=r2, lam=lam,
                    N_min=N_min, delta=delta, max_iters=max_iters, tol=tol)

    # 收集统计
    timings_runs = []  # list of list length K
    bounds_runs = []   # list of list length (K-1), 右侧边界 BMI 值

    for run_idx in range(n_runs):
        # 注入误差
        bmi = add_random_error(df['BMI'].to_numpy(), bmi_error_rate)
        # 保证 BMI 不越界太离谱（可选裁剪到原范围）
        bmi = np.clip(bmi, bmi_min0, bmi_max0)
        w = add_random_error(df['最早达标孕周'].to_numpy(), y_error_rate)
        # 孕周裁剪到 [10, 25]
        w = np.clip(w, 10.0, 25.0)

        # 依据 BMI 升序排序
        order = np.argsort(bmi)
        bmi_sorted = bmi[order]
        w_sorted = w[order]

        # 坐标下降优化
        try:
            boundaries, _, details = coordinate_descent(w_sorted=w_sorted, K=K, params=params)
        except Exception:
            # 若异常，跳过该轮
            continue

        # 每组 T* 与边界（右端 BMI）
        t_per_group = []
        right_edges = []
        n = len(bmi_sorted)
        idx_starts = [0] + [b + 1 for b in boundaries]
        idx_ends = boundaries + [n - 1]
        for det, s, e in zip(details, idx_starts, idx_ends):
            t_per_group.append(float(det['T_star']))
            right_edges.append(float(bmi_sorted[e]))

        if len(t_per_group) == K and len(right_edges) == K:
            timings_runs.append(t_per_group)
            bounds_runs.append(right_edges[:-1])  # 仅 K-1 个内部边界

    # 汇总统计
    if not timings_runs:
        raise RuntimeError('鲁棒性分析未得到有效结果，请减少 n_runs 或放宽参数。')

    timings_arr = np.array(timings_runs)  # (M, K)
    bounds_arr = np.array(bounds_runs)    # (M, K-1)

    # T* 统计
    timing_rows = []
    for g in range(K):
        vals = timings_arr[:, g]
        timing_rows.append({
            'group': g + 1,
            'n_runs': int(len(vals)),
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'q10': float(np.percentile(vals, 10)),
            'q50': float(np.percentile(vals, 50)),
            'q90': float(np.percentile(vals, 90)),
        })
    timing_df = pd.DataFrame(timing_rows)

    # 边界统计（右端点）
    bounds_rows = []
    for b in range(K - 1):
        vals = bounds_arr[:, b]
        bounds_rows.append({
            'boundary_index_between_group': f'{b+1}-{b+2}',
            'n_runs': int(len(vals)),
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'q10': float(np.percentile(vals, 10)),
            'q50': float(np.percentile(vals, 50)),
            'q90': float(np.percentile(vals, 90)),
        })
    bounds_df = pd.DataFrame(bounds_rows)

    os.makedirs(os.path.dirname(out_timing_csv), exist_ok=True)
    timing_df.to_csv(out_timing_csv, index=False, encoding='utf-8-sig')
    bounds_df.to_csv(out_bounds_csv, index=False, encoding='utf-8-sig')

    print('鲁棒性分析完成')
    print('\nT* 分布统计:')
    print(timing_df.to_string(index=False, float_format='%.3f'))
    print('\n边界分布统计:')
    print(bounds_df.to_string(index=False, float_format='%.3f'))
    print(f"\n已保存: {out_timing_csv}\n已保存: {out_bounds_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BMI 分段与最佳时点的鲁棒性分析（简洁版）')
    parser.add_argument('--processed', type=str, default='/Users/Mac/Downloads/mm/2/processed_data_problem2.csv')
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
    parser.add_argument('--y_error_rate', type=float, default=0.05)
    parser.add_argument('--bmi_error_rate', type=float, default=0.00)
    parser.add_argument('--n_runs', type=int, default=200)
    parser.add_argument('--out_timing', type=str, default='/Users/Mac/Downloads/mm/2/results/robustness_timing_summary.csv')
    parser.add_argument('--out_bounds', type=str, default='/Users/Mac/Downloads/mm/2/results/robustness_boundaries_summary.csv')
    args = parser.parse_args()

    run(
        processed_csv=args.processed,
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
        y_error_rate=args.y_error_rate,
        bmi_error_rate=args.bmi_error_rate,
        n_runs=args.n_runs,
        out_timing_csv=args.out_timing,
        out_bounds_csv=args.out_bounds,
    )
