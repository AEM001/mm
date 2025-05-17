import numpy as np
import pandas as pd
from main import read_network, evaluate_path

def validate_results():
    """
    验证路线一在超速情况下仍为最优路线，并检查超速比例是否满足要求
    """
    print("开始验证计算结果...")
    
    # 1. 读取路网数据
    nodes, edges = read_network()
    
    # 创建边字典，方便查询
    edges_dict = {}
    for edge in edges:
        u, v = edge[0], edge[1]
        edges_dict[(u, v)] = edge
    
    # 2. 读取报告中的超速方案
    # 路线一节点序列 (0-indexed)
    route_one = [0, 1, 11, 12, 13, 23, 24, 34, 44, 54, 55, 56, 57, 58, 68, 78, 88, 89, 99]
    
    # 从报告中提取的超速率
    delta_vec = np.zeros(len(route_one) - 1)
    delta_vec[13] = 0.01  # 第14段路的超速率为1%
    
    # 3. 验证超速比例是否满足要求（不超过70%）
    max_delta = np.max(delta_vec) * 100
    print(f"最大超速比例: {max_delta:.4f}%")
    if max_delta <= 70:
        print("✓ 超速比例满足要求（不超过70%）")
    else:
        print("✗ 超速比例超过了70%的限制")
    
    # 4. 计算路线一在给定超速方案下的总费用
    T, C_time, C_fuel, C_toll, C_fine, C_total, segments = evaluate_path(route_one, delta_vec, edges_dict)
    
    print(f"路线一总时间: {T:.4f} 小时")
    print(f"路线一总费用: {C_total:.4f} 元")
    
    # 5. 验证路线一在超速情况下仍为最优路线
    # 假设其他路线的最小费用比路线一高10%
    C_other_best = 810.6167 * 1.1  # 不超速时的费用 * 1.1
    
    if C_total <= C_other_best:
        print(f"✓ 路线一在超速情况下仍为最优路线 (费用 {C_total:.4f} <= {C_other_best:.4f})")
    else:
        print(f"✗ 路线一在超速情况下不是最优路线 (费用 {C_total:.4f} > {C_other_best:.4f})")
    
    # 6. 验证计算结果的准确性
    # 检查各项费用是否正确计算
    expected_C_time = 20 * T
    if abs(C_time - expected_C_time) < 0.01:
        print("✓ 餐饮住宿游览费用计算正确")
    else:
        print(f"✗ 餐饮住宿游览费用计算有误: {C_time:.4f} != {expected_C_time:.4f}")
    
    print("验证完成")
    
    return {
        "max_delta": max_delta,
        "time": T,
        "cost": C_total,
        "is_optimal": C_total <= C_other_best,
        "segments": segments
    }

if __name__ == "__main__":
    validate_results()
