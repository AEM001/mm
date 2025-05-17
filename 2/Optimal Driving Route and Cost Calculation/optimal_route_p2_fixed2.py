import numpy as np
import heapq
import matplotlib.pyplot as plt
from road_network_p2 import *
from cost_functions_p2 import *

# === 四）多标记Dijkstra算法 ===

def multi_label_dijkstra(adj, start, end, T_max):
    """
    多标记Dijkstra算法，在时间约束下寻找期望费用最小的路径
    
    参数:
        adj: 邻接表
        start: 起点索引
        end: 终点索引
        T_max: 最大时间约束（小时）
        
    返回:
        (最小期望费用, 路径, 超速方案)
    """
    print(f"开始多标记Dijkstra算法，从节点{start+1}到节点{end+1}，时间约束{T_max:.4f}小时")
    
    # 每个节点维护一组"(time, cost)"的Pareto前沿状态
    # labels[u] = [(time1, cost1, prev_node1, prev_delta1), (time2, cost2, prev_node2, prev_delta2), ...]
    labels = [[] for _ in range(100)]
    
    # 初始状态
    labels[start].append((0.0, 0.0, None, None))  # (time, cost, prev_node, prev_delta)
    
    # 优先队列：(cost, time, node, path_so_far, speed_plan_so_far)
    # 直接在队列中维护完整路径和超速方案，避免回溯错误
    pq = [(0.0, 0.0, start, [start], [])]
    
    # 记录最优解
    best_cost = float('inf')
    best_path = None
    best_speed_plan = None
    best_time = float('inf')
    
    # 记录处理过的状态数
    processed_states = 0
    
    # 记录到达终点的可行解数量
    feasible_solutions = 0
    
    while pq:
        cost_u, time_u, u, path_so_far, speed_plan_so_far = heapq.heappop(pq)
        processed_states += 1
        
        # 调试输出
        if processed_states % 10000 == 0:
            print(f"已处理{processed_states}个状态，当前队列长度：{len(pq)}")
        
        # 如果时间已经超出约束，跳过
        if time_u > T_max:
            continue
            
        # 如果已经找到更好的解，跳过
        if cost_u > best_cost:
            continue
        
        # 如果到达终点，更新最优解
        if u == end:
            if time_u <= T_max and (cost_u < best_cost or (cost_u == best_cost and time_u < best_time)):
                best_cost = cost_u
                best_time = time_u
                best_path = path_so_far.copy()
                best_speed_plan = speed_plan_so_far.copy()
                feasible_solutions += 1
                print(f"找到可行解 #{feasible_solutions}：费用={best_cost:.2f}元，时间={best_time:.4f}小时")
            continue
        
        # 遍历相邻路段
        for v, limit, vertical, has_fixed_radar in adj[u]:
            # 如果节点已经在当前路径中，跳过（避免环路）
            if v in path_so_far:
                continue
                
            # 尝试不同的超速选项
            for delta in speed_options:
                # 计算该路段的成本和时间
                seg_cost, seg_time = calculate_segment_cost(limit, delta, has_fixed_radar)
                
                # 计算新的累计时间和成本
                new_time = time_u + seg_time
                new_cost = cost_u + seg_cost
                
                # 如果超出时间约束，跳过
                if new_time > T_max:
                    continue
                
                # 检查是否被Pareto支配
                dominated = False
                for t, c, _, _ in labels[v]:
                    if t <= new_time and c <= new_cost:
                        dominated = True
                        break
                
                if dominated:
                    continue
                
                # 移除被新状态支配的旧状态
                labels[v] = [(t, c, p, d) for t, c, p, d in labels[v] 
                             if not (new_time <= t and new_cost <= c)]
                
                # 添加新状态
                labels[v].append((new_time, new_cost, u, delta))
                
                # 更新路径和超速方案
                new_path = path_so_far + [v]
                new_speed_plan = speed_plan_so_far + [delta]
                
                # 加入优先队列
                heapq.heappush(pq, (new_cost, new_time, v, new_path, new_speed_plan))
    
    print(f"算法结束，共处理{processed_states}个状态，找到{feasible_solutions}个可行解")
    
    # 如果没有找到可行解
    if best_path is None:
        print("未找到满足时间约束的可行路径！")
        return None, None, None
    
    print(f"最优路径长度：{len(best_path)}，超速方案长度：{len(best_speed_plan)}")
    
    return best_cost, best_path, best_speed_plan

# 运行算法
print("\n开始计算时间约束下的最优路径...")
start_node = 0  # 起点（甲地，左下角1号路口）
end_node = 99   # 终点（乙地，右上角100号路口）

min_cost, optimal_path, speed_plan = multi_label_dijkstra(adj, start_node, end_node, T_max)

if min_cost is None:
    print("未找到满足时间约束的可行路径！")
else:
    print(f"\n最小期望费用 C2 = {min_cost:.2f} 元")
    print(f"路径节点序列（原图编号）: {[index_to_number(idx) for idx in optimal_path]}")
    print(f"路径长度: {len(optimal_path)}个节点，{len(optimal_path)-1}段路")
    
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
                "限速": f"{limit} km/h",
                "超速比例": f"{delta*100:.0f}%",
                "实际速度": f"{actual_speed:.1f} km/h",
                "时间": f"{seg_time:.4f} 小时",
                "时间成本": f"{time_cost:.2f} 元",
                "油费": f"{fuel_cost:.2f} 元",
                "通行费": f"{toll_cost:.2f} 元",
                "期望罚款": f"{fine_cost:.2f} 元",
                "总费用": f"{seg_cost:.2f} 元"
            })
            
            total_distance += L
            total_time += seg_time
            total_cost += seg_cost
            total_fine += fine_cost
    
    # 打印路径详情
    print("\n路径详细信息:")
    print(f"{'起点':^15} | {'终点':^15} | {'限速':^8} | {'超速比例':^8} | {'实际速度':^10} | {'时间':^10} | {'时间成本':^8} | {'油费':^8} | {'通行费':^8} | {'期望罚款':^10} | {'总费用':^10}")
    print("-" * 140)
    
    for detail in path_details:
        print(f"{detail['起点']:^15} | {detail['终点']:^15} | {detail['限速']:^8} | {detail['超速比例']:^8} | {detail['实际速度']:^10} | {detail['时间']:^10} | {detail['时间成本']:^8} | {detail['油费']:^8} | {detail['通行费']:^8} | {detail['期望罚款']:^10} | {detail['总费用']:^10}")
    
    print("-" * 140)
    print(f"总计: {len(path_details)}段路径, 总距离 {total_distance} km, 总时间 {total_time:.4f} 小时, 总费用 {total_cost:.2f} 元")
    print(f"费用明细: 时间成本 {total_time*time_cost_rate:.2f} 元 + 油费和通行费 {total_cost-total_fine-total_time*time_cost_rate:.2f} 元 + 期望罚款 {total_fine:.2f} 元")
    
    # 检查时间约束
    print(f"\n时间约束检查: {total_time:.4f} 小时 <= {T_max:.4f} 小时: {'满足' if total_time <= T_max else '不满足'}")
    print(f"时间占比: {total_time/T_max*100:.2f}% (相对于时间预算)")
    print(f"时间占比: {total_time/(10.833)*100:.2f}% (相对于问题一中的T1)")

    # 可视化路径
    def visualize_path(path, speed_plan, title):
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
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.close()

    try:
        # 可视化最优路径
        visualize_path(optimal_path, speed_plan, "问题二最优路径与超速方案")
        print("\n已生成路径可视化图: 问题二最优路径与超速方案.png")
    except Exception as e:
        print(f"生成可视化图时出错: {e}")
        print("继续执行后续代码...")
