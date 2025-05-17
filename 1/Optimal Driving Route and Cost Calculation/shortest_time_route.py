import numpy as np
import heapq
import matplotlib.pyplot as plt
from road_network import *

# 计算时间最短路径
print("计算时间最短路径...")
start_node = 0  # 起点（甲地，左下角1号路口）
end_node = 99   # 终点（乙地，右上角100号路口）

# 使用Dijkstra算法计算最短时间路径
shortest_time, shortest_time_path = dijkstra(adj, start_node, end_node, weight_time)

# 输出结果
print(f"\n时间最短路径总时间 T1 = {shortest_time:.4f} 小时 (约 {int(shortest_time)}小时{int((shortest_time-int(shortest_time))*60)}分钟)")

# 计算路径详情
print("\n时间最短路径详细信息:")
print("节点序列（原图编号）:", [index_to_number(idx) for idx in shortest_time_path])

# 计算每段路径的详细信息
total_distance = 0
total_time = 0
path_details = []

for i in range(len(shortest_time_path) - 1):
    u = shortest_time_path[i]
    v = shortest_time_path[i+1]
    
    # 找到对应的边信息
    edge_info = None
    for neighbor, limit, vertical in adj[u]:
        if neighbor == v:
            edge_info = (limit, vertical)
            break
    
    if edge_info:
        limit, vertical = edge_info
        segment_time = weight_time(limit, vertical)
        direction = "纵向" if vertical else "横向"
        road_type = "高速公路" if limit == 120 else "普通公路"
        
        u_r, u_c = index_to_rc(u)
        v_r, v_c = index_to_rc(v)
        
        path_details.append({
            "起点": f"({u_r},{u_c}) - {index_to_number(u)}号路口",
            "终点": f"({v_r},{v_c}) - {index_to_number(v)}号路口",
            "方向": direction,
            "道路类型": road_type,
            "限速": f"{limit} km/h",
            "距离": f"{L} km",
            "时间": f"{segment_time:.4f} 小时"
        })
        
        total_distance += L
        total_time += segment_time

# 打印路径详情
print("\n路径详细信息:")
print(f"{'起点':^15} | {'终点':^15} | {'方向':^6} | {'道路类型':^8} | {'限速':^8} | {'距离':^6} | {'时间':^10}")
print("-" * 80)

for detail in path_details:
    print(f"{detail['起点']:^15} | {detail['终点']:^15} | {detail['方向']:^6} | {detail['道路类型']:^8} | {detail['限速']:^8} | {detail['距离']:^6} | {detail['时间']:^10}")

print("-" * 80)
print(f"总计: {len(path_details)}段路径, 总距离 {total_distance} km, 总时间 {total_time:.4f} 小时")

# 可视化路径
def visualize_path(path, title):
    plt.figure(figsize=(12, 12))
    
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
    
    # 绘制路径
    path_x = []
    path_y = []
    for node_idx in path:
        r, c = index_to_rc(node_idx)
        # 转换为坐标系（从0开始）
        y = r - 1
        x = c - 1
        path_x.append(x)
        path_y.append(y)
    
    plt.plot(path_x, path_y, 'r-', linewidth=2)
    plt.plot(path_x[0], path_y[0], 'go', markersize=12, label='起点')
    plt.plot(path_x[-1], path_y[-1], 'bo', markersize=12, label='终点')
    
    # 添加路径节点标签
    for i, node_idx in enumerate(path):
        node_num = index_to_number(node_idx)
        r, c = index_to_rc(node_idx)
        y = r - 1
        x = c - 1
        plt.text(x+0.1, y+0.1, f"{i+1}", color='red', fontsize=10, fontweight='bold')
    
    plt.title(title)
    plt.xlabel('列（从左到右）')
    plt.ylabel('行（从下到上）')
    plt.grid(True)
    plt.legend()
    plt.gca().invert_yaxis()  # 反转y轴，使得原点在左下角
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

# 可视化最短时间路径
visualize_path(shortest_time_path, "时间最短路径")
print("\n已生成路径可视化图: 时间最短路径.png")
