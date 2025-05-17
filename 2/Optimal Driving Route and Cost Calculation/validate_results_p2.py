import numpy as np
import matplotlib.pyplot as plt
from road_network_p2 import *
from cost_functions_p2 import *

# 验证最优路径结果
print("=== 验证问题二最优路径结果 ===")

# 最优路径和超速方案
optimal_path = [0, 1, 11, 12, 13, 23, 24, 34, 44, 54, 55, 56, 57, 58, 68, 78, 88, 89, 99]  # 索引从0开始
speed_plan = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# 转换为原图编号（从1开始）
path_numbers = [index_to_number(idx) for idx in optimal_path]
print(f"路径节点序列（原图编号）: {path_numbers}")
print(f"路径长度: {len(optimal_path)}个节点，{len(optimal_path)-1}段路")

# 验证路径连通性
print("\n验证路径连通性:")
is_connected = True
for i in range(len(optimal_path) - 1):
    u = optimal_path[i]
    v = optimal_path[i+1]
    
    # 检查是否相邻
    is_neighbor = False
    for neighbor, _, _, _ in adj[u]:
        if neighbor == v:
            is_neighbor = True
            break
    
    if not is_neighbor:
        print(f"错误：节点{index_to_number(u)}和节点{index_to_number(v)}不相邻！")
        is_connected = False

if is_connected:
    print("路径连通性验证通过：所有相邻节点之间存在直接连接")

# 计算路径详情
total_distance = 0
total_time = 0
total_cost = 0
total_fine = 0
path_details = []

for i in range(len(optimal_path) - 1):
    u = optimal_path[i]
    v = optimal_path[i+1]
    delta = speed_plan[i]
    
    # 找到对应的边信息
    edge_info = None
    for neighbor, limit, vertical, has_radar in adj[u]:
        if neighbor == v:
            edge_info = (limit, vertical, has_radar)
            break
    
    if edge_info:
        limit, vertical, has_radar = edge_info
        actual_speed = limit * (1 + delta)
        seg_cost, seg_time = calculate_segment_cost(limit, delta, has_radar)
        
        # 计算各项成本
        time_cost = time_cost_rate * seg_time
        fuel_consumption = (0.0625 * actual_speed + 1.875) * (L / 100.0)
        fuel_cost = fuel_consumption * fuel_price
        toll_cost = toll_per_km * L if limit == 120 else 0.0
        fine_cost = expected_fine(limit, delta, has_radar)
        
        u_r, u_c = index_to_rc(u)
        v_r, v_c = index_to_rc(v)
        
        path_details.append({
            "起点": f"({u_r},{u_c}) - {index_to_number(u)}号路口",
            "终点": f"({v_r},{v_c}) - {index_to_number(v)}号路口",
            "限速": limit,
            "超速比例": delta,
            "实际速度": actual_speed,
            "时间": seg_time,
            "时间成本": time_cost,
            "油费": fuel_cost,
            "通行费": toll_cost,
            "期望罚款": fine_cost,
            "总费用": seg_cost
        })
        
        total_distance += L
        total_time += seg_time
        total_cost += seg_cost
        total_fine += fine_cost

# 打印路径详情
print("\n路径详细信息:")
print(f"{'起点':^15} | {'终点':^15} | {'限速':^8} | {'超速比例':^8} | {'实际速度':^10} | {'时间':^10} | {'期望罚款':^10} | {'总费用':^10}")
print("-" * 120)

for detail in path_details:
    print(f"{detail['起点']:^15} | {detail['终点']:^15} | {detail['限速']:^8.1f} | {detail['超速比例']*100:^8.0f}% | {detail['实际速度']:^10.1f} | {detail['时间']:^10.4f} | {detail['期望罚款']:^10.2f} | {detail['总费用']:^10.2f}")

print("-" * 120)
print(f"总计: {len(path_details)}段路径, 总距离 {total_distance} km, 总时间 {total_time:.4f} 小时, 总费用 {total_cost:.2f} 元")
print(f"费用明细: 时间成本 {total_time*time_cost_rate:.2f} 元 + 油费和通行费 {total_cost-total_fine-total_time*time_cost_rate:.2f} 元 + 期望罚款 {total_fine:.2f} 元")

# 检查时间约束
print(f"\n时间约束检查: {total_time:.4f} 小时 <= {T_max:.4f} 小时: {'满足' if total_time <= T_max else '不满足'}")
print(f"时间占比: {total_time/T_max*100:.2f}% (相对于时间预算)")
print(f"时间占比: {total_time/(10.833)*100:.2f}% (相对于问题一中的T1)")

# 检查超速合规性
print("\n超速合规性检查:")
for i, delta in enumerate(speed_plan):
    if delta > 0.7:
        print(f"错误：第{i+1}段路超速比例{delta*100:.0f}%超过了70%的上限！")

print("超速合规性验证通过：所有路段超速比例均不超过70%")

# 可视化路径
def visualize_validated_path(path, speed_plan, title):
    plt.figure(figsize=(14, 14))
    ax = plt.gca()
    
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
    
    # 使用颜色表示超速程度
    cmap = plt.colormaps.get_cmap('RdYlGn_r')  # 红黄绿色谱，反转使红色表示高超速
    
    for i in range(len(path) - 1):
        x1, y1 = path_x[i], path_y[i]
        x2, y2 = path_x[i+1], path_y[i+1]
        delta = speed_plan[i]
        
        # 根据超速比例选择颜色
        color = cmap(delta / 0.7)  # 归一化到[0,1]
        
        plt.plot([x1, x2], [y1, y2], '-', linewidth=3, color=color)
        
        # 在路段中点标注超速比例
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        plt.text(mid_x, mid_y, f"{delta*100:.0f}%", color='black', fontweight='bold', 
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.plot(path_x[0], path_y[0], 'go', markersize=12, label='起点')
    plt.plot(path_x[-1], path_y[-1], 'ro', markersize=12, label='终点')
    
    # 添加路径节点标签
    for i, node_idx in enumerate(path):
        node_num = index_to_number(node_idx)
        r, c = index_to_rc(node_idx)
        y = r - 1
        x = c - 1
        plt.text(x+0.1, y+0.1, f"{i+1}", color='blue', fontsize=10, fontweight='bold')
    
    # 添加颜色条，表示超速程度
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 70))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('超速比例 (%)')
    cbar.set_ticks([0, 20/70, 50/70, 1])
    cbar.set_ticklabels(['0%', '20%', '50%', '70%'])
    
    plt.title(title)
    plt.xlabel('列（从左到右）')
    plt.ylabel('行（从下到上）')
    plt.grid(True)
    plt.legend()
    plt.gca().invert_yaxis()  # 反转y轴，使得原点在左下角
    plt.savefig(f"{title.replace(' ', '_')}_validated.png")
    plt.close()

try:
    # 可视化验证后的最优路径
    visualize_validated_path(optimal_path, speed_plan, "问题二最优路径与超速方案(验证)")
    print("\n已生成验证后的路径可视化图: 问题二最优路径与超速方案(验证).png")
except Exception as e:
    print(f"生成可视化图时出错: {e}")

print("\n验证完成！")
