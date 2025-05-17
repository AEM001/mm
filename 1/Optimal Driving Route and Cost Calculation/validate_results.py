import numpy as np
import matplotlib.pyplot as plt
from road_network import *

# 验证路径和计算结果
print("开始验证路径和计算结果...")

# 加载两条路径的结果
from shortest_time_route import shortest_time, shortest_time_path
from minimum_cost_route import min_cost, min_cost_path

# 验证函数
def validate_path(path, adj):
    """验证路径的连续性"""
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        
        # 检查v是否是u的邻居
        is_neighbor = False
        for neighbor, _, _ in adj[u]:
            if neighbor == v:
                is_neighbor = True
                break
        
        if not is_neighbor:
            return False, f"路径不连续: 节点{index_to_number(u)}与节点{index_to_number(v)}不相邻"
    
    return True, "路径连续性验证通过"

# 验证时间计算
def validate_time_calculation(path, adj):
    """验证时间计算的准确性"""
    total_time = 0
    
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        
        # 找到对应的边信息
        edge_info = None
        for neighbor, limit, vertical in adj[u]:
            if neighbor == v:
                edge_info = (limit, vertical)
                break
        
        if edge_info:
            limit, vertical = edge_info
            segment_time = weight_time(limit, vertical)
            total_time += segment_time
    
    return total_time, abs(total_time - shortest_time) < 1e-6

# 验证费用计算
def validate_cost_calculation(path, adj):
    """验证费用计算的准确性"""
    total_cost = 0
    
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        
        # 找到对应的边信息
        edge_info = None
        for neighbor, limit, vertical in adj[u]:
            if neighbor == v:
                edge_info = (limit, vertical)
                break
        
        if edge_info:
            limit, vertical = edge_info
            segment_cost = weight_cost(limit, vertical)
            total_cost += segment_cost
    
    return total_cost, abs(total_cost - min_cost) < 1e-6

# 验证最短时间路径
print("\n验证最短时间路径:")
time_path_continuous, time_path_msg = validate_path(shortest_time_path, adj)
print(f"1. 路径连续性: {'通过' if time_path_continuous else '失败'} - {time_path_msg}")

recalculated_time, time_calculation_correct = validate_time_calculation(shortest_time_path, adj)
print(f"2. 时间计算: {'通过' if time_calculation_correct else '失败'} - 重新计算得到 {recalculated_time:.4f} 小时，原计算为 {shortest_time:.4f} 小时")

# 验证最少费用路径
print("\n验证最少费用路径:")
cost_path_continuous, cost_path_msg = validate_path(min_cost_path, adj)
print(f"1. 路径连续性: {'通过' if cost_path_continuous else '失败'} - {cost_path_msg}")

recalculated_cost, cost_calculation_correct = validate_cost_calculation(min_cost_path, adj)
print(f"2. 费用计算: {'通过' if cost_calculation_correct else '失败'} - 重新计算得到 {recalculated_cost:.2f} 元，原计算为 {min_cost:.2f} 元")

# 验证路径起点和终点
print("\n验证路径起点和终点:")
print(f"1. 最短时间路径: 起点为节点{index_to_number(shortest_time_path[0])}，终点为节点{index_to_number(shortest_time_path[-1])}")
print(f"2. 最少费用路径: 起点为节点{index_to_number(min_cost_path[0])}，终点为节点{index_to_number(min_cost_path[-1])}")

# 比较两条路径
print("\n比较两条路径:")
print(f"1. 最短时间路径: {len(shortest_time_path)}个节点, 总时间 {shortest_time:.4f} 小时, 总费用 {validate_cost_calculation(shortest_time_path, adj)[0]:.2f} 元")
print(f"2. 最少费用路径: {len(min_cost_path)}个节点, 总时间 {validate_time_calculation(min_cost_path, adj)[0]:.4f} 小时, 总费用 {min_cost:.2f} 元")

# 可视化两条路径的对比
def visualize_comparison(time_path, cost_path):
    plt.figure(figsize=(14, 14))
    
    # 绘制网格
    for i in range(11):
        plt.axhline(y=i, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=i, color='gray', linestyle='-', alpha=0.3)
    
    # 绘制所有节点
    for i in range(10):
        for j in range(10):
            node_idx = i * 10 + j
            node_num = index_to_number(node_idx)
            plt.plot(j, i, 'o', markersize=8, color='lightgray')
            plt.text(j, i, str(node_num), ha='center', va='center', fontsize=8)
    
    # 绘制最短时间路径
    time_path_x = []
    time_path_y = []
    for node_idx in time_path:
        r, c = index_to_rc(node_idx)
        y = r - 1
        x = c - 1
        time_path_x.append(x)
        time_path_y.append(y)
    
    plt.plot(time_path_x, time_path_y, 'r-', linewidth=2, label='时间最短路径')
    
    # 绘制最少费用路径
    cost_path_x = []
    cost_path_y = []
    for node_idx in cost_path:
        r, c = index_to_rc(node_idx)
        y = r - 1
        x = c - 1
        cost_path_x.append(x)
        cost_path_y.append(y)
    
    plt.plot(cost_path_x, cost_path_y, 'b--', linewidth=2, label='费用最少路径')
    
    # 标记起点和终点
    plt.plot(time_path_x[0], time_path_y[0], 'go', markersize=12, label='起点')
    plt.plot(time_path_x[-1], time_path_y[-1], 'mo', markersize=12, label='终点')
    
    plt.title('路径对比图')
    plt.xlabel('列（从左到右）')
    plt.ylabel('行（从下到上）')
    plt.grid(True)
    plt.legend()
    plt.gca().invert_yaxis()  # 反转y轴，使得原点在左下角
    plt.savefig("路径对比图.png")
    plt.close()

# 可视化两条路径的对比
visualize_comparison(shortest_time_path, min_cost_path)
print("\n已生成路径对比可视化图: 路径对比图.png")

print("\n验证结论:")
if time_path_continuous and time_calculation_correct and cost_path_continuous and cost_calculation_correct:
    print("所有验证通过！计算结果可信。")
    print(f"时间最短路径总时间 T1 = {shortest_time:.4f} 小时 (约 {int(shortest_time)}小时{int((shortest_time-int(shortest_time))*60)}分钟)")
    print(f"费用最少路径总费用 C1 = {min_cost:.2f} 元")
else:
    print("验证未通过，请检查计算过程。")
