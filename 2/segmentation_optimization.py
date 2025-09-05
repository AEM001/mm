#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMI 分组边界的局部坐标下降优化（简洁版）
- 数据来源：/Users/Mac/Downloads/mm/2/processed_data_problem2.csv
- 输入字段：BMI, 最早达标孕周
- 思路：
  1) 按 BMI 升序，初始化为 K 个等人数分组（满足每组 >= N_min）
  2) 迭代地对每个边界做小幅左右移动（±delta 样本），若总目标下降则接受
  3) 组内最佳时点 T* = max( q_c(w), min_week )，覆盖率约束 cov >= c
  4) 目标 J = ∑_g [ N_g * timeRisk(T_g) + lam * ∑_i max(0, T_g - w_i) ]
- 输出：/Users/Mac/Downloads/mm/2/results/segment_optimized_summary.csv
"""

import os
import argparse
import numpy as np
import pandas as pd


# ------------------------ 基本代价与工具函数 ------------------------ #

def compute_T_star(w_segment: np.ndarray, c: float = 0.90, min_week: float = 12.0) -> float:
    """给定一段 w（最早达标孕周），返回满足覆盖率的最优检测时点 T*
    - 简化：T* 取 max(q_c, min_week)
    """
    if len(w_segment) == 0:
        return float('inf')
    # 使用上分位的秩统计量，确保经验覆盖率 cov >= c
    n_seg = len(w_segment)
    k = max(1, int(np.ceil(c * n_seg)))  # 第 k 个有序样本（1-based）
    q_c = float(np.sort(w_segment)[k - 1])
    return max(q_c, float(min_week))


def time_risk(T: float, r1: float = 1.0, r2: float = 3.0) -> float:
    """时间风险分段函数
    - T <= 12: 0
    - 12 < T <= 27: r1
    - T > 27: r2
    """
    if T <= 12.0:
        return 0.0
    if T <= 27.0:
        return float(r1)
    return float(r2)


def segment_cost(w_segment: np.ndarray,
                 c: float = 0.90,
                 min_week: float = 12.0,
                 r1: float = 1.0,
                 r2: float = 3.0,
                 lam: float = 0.0):
    """计算一个分段的最优时点与代价
    返回 (T_star, cost, coverage, delay_penalty)
    """
    if len(w_segment) == 0:
        return float('inf'), float('inf'), 0.0, 0.0
    T_star = compute_T_star(w_segment, c=c, min_week=min_week)
    # 覆盖率（实际）
    coverage = float(np.mean(w_segment <= T_star)) if len(w_segment) > 0 else 0.0
    # 延迟惩罚
    delay = np.maximum(0.0, T_star - w_segment).sum()
    # 时间风险（按组样本数加权）
    trisk = len(w_segment) * time_risk(T_star, r1=r1, r2=r2)
    cost = trisk + lam * delay
    return float(T_star), float(cost), coverage, float(delay)


def total_cost(boundaries, w_sorted, params):
    """计算当前边界下的总目标与各组摘要
    boundaries: 升序切分点索引列表（长度 K-1），定义分段 [0..b0], [b0+1..b1], ..., [b_{K-2}+1..n-1]
    返回 total_cost, details(list of dict)
    """
    n = len(w_sorted)
    idx_starts = [0] + [b + 1 for b in boundaries]
    idx_ends = boundaries + [n - 1]
    details = []
    total = 0.0
    for g, (s, e) in enumerate(zip(idx_starts, idx_ends), start=1):
        w_seg = w_sorted[s:e+1]
        T_star, cst, cov, delay = segment_cost(
            w_seg,
            c=params.c,
            min_week=params.min_week,
            r1=params.r1,
            r2=params.r2,
            lam=params.lam,
        )
        total += cst
        details.append({
            'group': g,
            'start_idx': s,
            'end_idx': e,
            'n': len(w_seg),
            'T_star': T_star,
            'coverage': cov,
            'delay_sum': delay,
            'cost': cst,
        })
    return float(total), details


def initial_equal_boundaries(n: int, K: int, N_min: int):
    """构造均匀分 K 段的初始边界，保证每段 >= N_min
    返回边界索引列表（长度 K-1）。若无法满足返回 None
    """
    if n < K * N_min:
        return None
    # 基本等分
    base = n // K
    rem = n % K
    sizes = [base + (1 if i < rem else 0) for i in range(K)]
    # 若有段 < N_min，则从其他段借样本
    for i in range(K):
        if sizes[i] < N_min:
            need = N_min - sizes[i]
            # 从左右相邻段借
            for j in range(K):
                if j == i:
                    continue
                can = max(0, sizes[j] - N_min)
                take = min(can, need)
                sizes[j] -= take
                sizes[i] += take
                need -= take
                if need == 0:
                    break
            if need > 0:
                return None  # 无法满足
    # 累积得到边界
    boundaries = []
    acc = 0
    for k in range(K - 1):
        acc += sizes[k]
        boundaries.append(acc - 1)  # 以段末索引作为边界
    return boundaries


# ------------------------ 局部坐标下降优化 ------------------------ #

class Params:
    def __init__(self, c=0.90, min_week=12.0, r1=1.0, r2=3.0, lam=0.0, N_min=20, delta=3, max_iters=200, tol=1e-4):
        self.c = float(c)
        self.min_week = float(min_week)
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.lam = float(lam)
        self.N_min = int(N_min)
        self.delta = int(delta)
        self.max_iters = int(max_iters)
        self.tol = float(tol)


def valid_boundaries(boundaries, n, K, N_min):
    """检查边界合法性：严格递增、每段 >= N_min"""
    if len(boundaries) != K - 1:
        return False
    if not all(boundaries[i] < boundaries[i+1] for i in range(len(boundaries) - 1)):
        return False
    # 段长度
    idx_starts = [0] + [b + 1 for b in boundaries]
    idx_ends = boundaries + [n - 1]
    sizes = [e - s + 1 for s, e in zip(idx_starts, idx_ends)]
    return all(sz >= N_min for sz in sizes)


def coordinate_descent(w_sorted: np.ndarray,
                        K: int,
                        params: Params,
                        init_boundaries=None):
    """局部坐标下降优化 BMI 分段（在 w_sorted 上）
    - 返回 best_boundaries, best_total_cost, best_details
    """
    n = len(w_sorted)
    if n < K * params.N_min:
        raise ValueError(f"样本数不足以分为 {K} 组（每组至少 {params.N_min} 人）：n={n}")

    if init_boundaries is None:
        boundaries = initial_equal_boundaries(n, K, params.N_min)
        if boundaries is None:
            raise ValueError("无法构造满足 N_min 的初始边界")
    else:
        boundaries = init_boundaries[:]
        if not valid_boundaries(boundaries, n, K, params.N_min):
            raise ValueError("提供的初始边界不合法")

    best_total, best_details = total_cost(boundaries, w_sorted, params)

    for it in range(params.max_iters):
        improved = False
        # 依次尝试每个边界
        for bi in range(len(boundaries)):
            current_pos = boundaries[bi]
            # 搜索候选移动（-delta..-1, +1..+delta）
            for step in list(range(-params.delta, 0)) + list(range(1, params.delta + 1)):
                new_pos = current_pos + step
                cand = boundaries[:]
                cand[bi] = new_pos
                if not valid_boundaries(cand, n, K, params.N_min):
                    continue
                cand_total, cand_details = total_cost(cand, w_sorted, params)
                if cand_total + params.tol < best_total:  # 有效改进
                    boundaries = cand
                    best_total, best_details = cand_total, cand_details
                    improved = True
                    break  # 接受首次改进，进入下一个边界
            # 若已改进，边界集合发生变化，继续下一边界
        if not improved:
            # 一轮无改进，终止
            break
    return boundaries, best_total, best_details


# ------------------------ 主流程与结果输出 ------------------------ #

def run(input_csv: str,
        output_csv: str,
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
        assignments_out: str = ""):
    # 读取数据
    df = pd.read_csv(input_csv)
    if not {'BMI', '最早达标孕周'}.issubset(df.columns):
        raise ValueError("输入数据缺少必要列：'BMI', '最早达标孕周'")

    # 清理与排序
    df = df[['BMI', '最早达标孕周']].dropna().copy()
    df = df.sort_values('BMI').reset_index(drop=True)
    bmi = df['BMI'].to_numpy()
    w = df['最早达标孕周'].to_numpy()

    params = Params(c=c, min_week=min_week, r1=r1, r2=r2, lam=lam, N_min=N_min, delta=delta, max_iters=max_iters, tol=tol)

    # 运行坐标下降
    boundaries, best_total, best_details = coordinate_descent(w_sorted=w, K=K, params=params)

    # 组装摘要并输出
    n = len(w)
    idx_starts = [0] + [b + 1 for b in boundaries]
    idx_ends = boundaries + [n - 1]

    rows = []
    for det, s, e in zip(best_details, idx_starts, idx_ends):
        bmi_min = float(bmi[s])
        bmi_max = float(bmi[e])
        interval = f"[{bmi_min:.1f}, {bmi_max:.1f}]"
        rows.append({
            '组号': det['group'],
            '样本数': det['n'],
            'BMI区间': interval,
            'BMI范围': f"{bmi_min:.1f}-{bmi_max:.1f}",
            'BMI均值': float(bmi[s:e+1].mean()),
            'T_star(周)': det['T_star'],
            '覆盖率': det['coverage'],
            '延迟惩罚和': det['delay_sum'],
            '组内代价': det['cost'],
        })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values('组号')

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("优化完成：")
    print(out_df.to_string(index=False, float_format='%.3f'))
    print(f"\n总目标值 J = {best_total:.3f}")
    print(f"结果已保存至: {output_csv}")

    # 可选：保存每个样本的分组与T*明细，便于可视化与后续分析
    if assignments_out:
        assign_rows = []
        for det, s, e in zip(best_details, idx_starts, idx_ends):
            g = det['group']
            g_bmi_min = float(bmi[s])
            g_bmi_max = float(bmi[e])
            T_star_g = float(det['T_star'])
            for idx in range(s, e + 1):
                assign_rows.append({
                    'idx_sorted': idx,
                    'BMI': float(bmi[idx]),
                    '最早达标孕周': float(w[idx]),
                    'group': g,
                    'group_bmi_min': g_bmi_min,
                    'group_bmi_max': g_bmi_max,
                    'T_star': T_star_g,
                })
        assign_df = pd.DataFrame(assign_rows)
        os.makedirs(os.path.dirname(assignments_out), exist_ok=True)
        assign_df.to_csv(assignments_out, index=False, encoding='utf-8-sig')
        print(f"样本分组明细已保存至: {assignments_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BMI 分组边界的局部坐标下降优化（简洁版）')
    parser.add_argument('--input', type=str, default='/Users/Mac/Downloads/mm/2/processed_data_problem2.csv')
    parser.add_argument('--output', type=str, default='/Users/Mac/Downloads/mm/2/results/segment_optimized_summary.csv')
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
    parser.add_argument('--assignments', type=str, default='/Users/Mac/Downloads/mm/2/results/segment_assignments.csv',
                        help='每个样本的分组与T*明细输出路径，留空表示不保存')
    args = parser.parse_args()

    run(
        input_csv=args.input,
        output_csv=args.output,
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
        assignments_out=args.assignments,
    )
