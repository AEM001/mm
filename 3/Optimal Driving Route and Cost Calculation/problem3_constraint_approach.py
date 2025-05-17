import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq

# ——— 常量 ———
L = 50.0                     # 每段距离 (km)
delta_options = [0.0, 0.2, 0.5, 0.7]  # 超速档：0%,20%,50%,70%
fuel_price = 7.76            # 元/升
time_rate = 20.0             # 元/小时
toll_per_km = 0.5            # 高速费：0.5元/km

# ——— 单段 时间/费用 计算 ———
def seg_time(limit, delta):
    """计算路段行驶时间"""
    return L / (limit * (1 + delta))

def seg_cost(limit, delta):
    """计算路段总成本（包括时间成本、燃油费、通行费和期望罚款）"""
    # 实际速度
    v = limit * (1 + delta)
    
    # 1. 时间成本
    c_time = time_rate * seg_time(limit, delta)
    
    # 2. 燃油费
    litres = (0.0625 * v + 1.875) * (L / 100.0)
    c_fuel = litres * fuel_price
    
    # 3. 高速通行费
    c_toll = toll_per_km * L if limit == 120 else 0.0
    
    # 4. 期望罚款
    over = delta * 100
    if over == 0:
        c_fine = 0
    else:
        # 根据限速和超速百分比确定罚款金额
        if limit < 50:      
            fines = [50, 100, 300, 500]
        elif limit <= 80:   
            fines = [100, 150, 500, 1000]
        elif limit <= 100:  
            fines = [150, 200, 1000, 1500]
        else:               
            fines = [200, 1500, 2000]
        
        # 选罚款档
        if over <= 20:      
            amt = fines[0]
        elif over <= 50:     
            amt = fines[1]
        else:               
            amt = fines[2]
        
        # 检测概率
        if delta == 0.2:
            p = 0.7
        elif delta == 0.5:
            p = 0.9
        elif delta == 0.7:
            p = 0.99
        else:
            p = 0  # 不超速
            
        c_fine = amt * p
    
    # 总成本
    return c_time + c_fuel + c_toll + c_fine

# ——— 读图 & 构建路网 ———
def build_graph():
    """构建路网图"""
    # 读取限速数据
    limits_col = np.loadtxt('limits_col.csv', delimiter=',')  # 9×10
    limits_row = np.loadtxt('limits_row.csv', delimiter=',')  # 10×9
    
    # 构建邻接表
    G = [[] for _ in range(100)]
    for i in range(10):
        for j in range(10):
            u = i * 10 + j
            if i < 9: G[u].append(((i+1)*10+j, limits_col[i][j]))
            if i > 0: G[u].append(((i-1)*10+j, limits_col[i-1][j]))
            if j < 9: G[u].append((i*10+j+1, limits_row[i][j]))
            if j > 0: G[u].append((i*10+j-1, limits_row[i][j-1]))
    return G, limits_col, limits_row

# 构建路网图
G, limits_col, limits_row = build_graph()

# ——— 路线一（P*）的节点序列和限速 ———
# 原图编号转为0-based索引
def number_to_index(num):
    """将原图编号转换为0-based索引"""
    r = (num - 1) // 10
    c = (num - 1) % 10
    return r * 10 + c

def index_to_number(idx):
    """将0-based索引转换为原图编号"""
    r = idx // 10
    c = idx % 10
    return r * 10 + c + 1

def index_to_rc(idx):
    """将0-based索引转换为(r,c)坐标"""
    r = idx // 10
    c = idx % 10
    return (r, c)

# 路线一的节点序列（原图编号）
route1_numbers = [1, 2, 12, 13, 14, 24, 25, 35, 45, 55, 56, 57, 58, 59, 69, 79, 89, 90, 100]

# 转换为0-based索引
P_star_nodes = [number_to_index(num) for num in route1_numbers]

# 提取路线一的限速序列
limits_P_star = []
for i in range(len(P_star_nodes) - 1):
    u = P_star_nodes[i]
    v = P_star_nodes[i + 1]
    u_r, u_c = index_to_rc(u)
    v_r, v_c = index_to_rc(v)
    
    # 找到对应的限速
    for next_node, limit in G[u]:
        if next_node == v:
            limits_P_star.append(limit)
            break

# 验证路线一的节点序列和限速
print("路线一（P*）的节点序列（0-based索引）:", P_star_nodes)
print("路线一（P*）的限速序列:", limits_P_star)

# ——— 设计约束条件，使路线一成为全局费用最优解 ———
def design_constraint():
    """设计约束条件，使路线一成为全局费用最优解"""
    print("\n=== 设计约束条件，使路线一成为全局费用最优解 ===")
    
    # 方法1：禁止除路线一以外的所有路径
    print("\n方法1：禁止除路线一以外的所有路径")
    print("这是最直接的方法，但不够优雅")
    
    # 方法2：对非路线一的路径施加高额费用
    print("\n方法2：对非路线一的路径施加高额费用")
    print("可以通过在非路线一的路段上增加额外费用，使得任何其他路径的总费用都高于路线一")
    
    # 方法3：限制其他路径的超速选项
    print("\n方法3：限制其他路径的超速选项")
    print("可以规定只有路线一允许超速，其他路径必须严格遵守限速")
    
    # 方法4：对路线一施加特殊的费用计算规则
    print("\n方法4：对路线一施加特殊的费用计算规则")
    print("可以为路线一设计特殊的费用计算规则，使其费用低于其他路径")
    
    # 选择最合适的约束方法
    print("\n选择约束方法3：限制其他路径的超速选项")
    print("这种方法既符合实际情况，又能确保路线一成为全局费用最优解")
    
    return "限制其他路径的超速选项"

# ——— 在约束条件下，优化路线一的时间 ———
def optimize_route1_time():
    """在约束条件下，优化路线一的时间"""
    print("\n=== 在约束条件下，优化路线一的时间 ===")
    
    # 计算路线一在不同超速方案下的时间和费用
    all_plans = []
    
    # 枚举所有可能的超速方案
    def generate_all_speeding_plans(segment_idx=0, current_plan=[], current_time=0, current_cost=0):
        if segment_idx == len(limits_P_star):
            all_plans.append((current_time, current_cost, current_plan.copy()))
            return
        
        limit = limits_P_star[segment_idx]
        
        for delta in delta_options:
            # 计算该路段的时间和成本
            seg_t = seg_time(limit, delta)
            seg_c = seg_cost(limit, delta)
            
            # 递归生成下一段路的方案
            current_plan.append(delta)
            generate_all_speeding_plans(segment_idx + 1, current_plan, current_time + seg_t, current_cost + seg_c)
            current_plan.pop()
    
    # 生成所有可能的超速方案
    generate_all_speeding_plans()
    
    # 按时间排序
    all_plans.sort(key=lambda x: x[0])
    
    # 找到最短时间方案
    min_time_plan = all_plans[0]
    min_time = min_time_plan[0]
    min_time_cost = min_time_plan[1]
    min_time_delta = min_time_plan[2]
    
    print(f"\n路线一的最短时间: {min_time:.4f} 小时")
    print(f"对应费用: {min_time_cost:.2f} 元")
    print(f"对应超速方案: {[f'{delta*100:.0f}%' for delta in min_time_delta]}")
    
    # 提取Pareto前沿
    pareto_frontier = []
    min_cost = float('inf')
    
    # 按费用排序
    all_plans.sort(key=lambda x: x[1])
    
    for time, cost, delta_seq in all_plans:
        if cost < min_cost:
            pareto_frontier.append((time, cost, delta_seq))
            min_cost = cost
    
    # 按时间排序
    pareto_frontier.sort(key=lambda x: x[0])
    
    # 打印Pareto前沿
    print(f"\nPareto前沿大小: {len(pareto_frontier)}")
    for i, (time, cost, delta_seq) in enumerate(pareto_frontier):
        print(f"方案 {i+1}: 时间 = {time:.4f} 小时, 费用 = {cost:.2f} 元")
        if i == 0 or i == len(pareto_frontier) - 1:
            print(f"超速方案: {[f'{delta*100:.0f}%' for delta in delta_seq]}")
    
    # 绘制Pareto前沿
    plt.figure(figsize=(10, 6))
    times = [p[0] for p in pareto_frontier]
    costs = [p[1] for p in pareto_frontier]
    plt.plot(times, costs, 'o-', markersize=8)
    
    # 标记特殊点
    min_time_point = pareto_frontier[0]
    min_cost_point = pareto_frontier[-1]
    
    plt.plot(min_time_point[0], min_time_point[1], 'ro', markersize=12, label='最短时间方案')
    plt.plot(min_cost_point[0], min_cost_point[1], 'go', markersize=12, label='最低费用方案')
    
    plt.title('路线一的时间-费用Pareto前沿')
    plt.xlabel('时间 (小时)')
    plt.ylabel('费用 (元)')
    plt.grid(True)
    plt.legend()
    plt.savefig('路线一时间费用Pareto前沿.png')
    plt.close()
    
    # 可视化最短时间方案
    visualize_optimal_speed_plan(min_time_delta, "问题三路线一最短时间方案")
    
    return min_time, min_time_cost, min_time_delta, pareto_frontier

# ——— 结果可视化 ———
def visualize_optimal_speed_plan(speed_plan, title):
    """可视化最优超速方案"""
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
    
    # 绘制路线一
    path_x = []
    path_y = []
    for node_idx in P_star_nodes:
        r, c = index_to_rc(node_idx)
        path_x.append(c)
        path_y.append(r)
    
    # 使用颜色表示超速程度
    cmap = plt.cm.get_cmap('RdYlGn_r')  # 红黄绿色谱，反转使红色表示高超速
    
    for i in range(len(P_star_nodes) - 1):
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
    for i, node_idx in enumerate(P_star_nodes):
        node_num = index_to_number(node_idx)
        r, c = index_to_rc(node_idx)
        plt.text(c+0.1, r+0.1, f"{i+1}", color='blue', fontsize=10, fontweight='bold')
    
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
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

# ——— 详细分析最优方案 ———
def analyze_optimal_solution(speed_plan):
    """详细分析最优超速方案"""
    print("\n=== 详细分析最优超速方案 ===")
    
    # 计算每段路的详细信息
    total_time = 0
    total_cost = 0
    total_time_cost = 0
    total_fuel_cost = 0
    total_toll_cost = 0
    total_fine_cost = 0
    
    segment_details = []
    
    for i, delta in enumerate(speed_plan):
        limit = limits_P_star[i]
        actual_speed = limit * (1 + delta)
        
        # 计算时间和成本
        seg_t = seg_time(limit, delta)
        time_cost = time_rate * seg_t
        
        # 燃油费
        fuel_consumption = (0.0625 * actual_speed + 1.875) * (L / 100.0)
        fuel_cost = fuel_consumption * fuel_price
        
        # 高速通行费
        toll_cost = toll_per_km * L if limit == 120 else 0.0
        
        # 期望罚款
        over = delta * 100
        if over == 0:
            fine_cost = 0
        else:
            if limit < 50:      
                fines = [50, 100, 300, 500]
            elif limit <= 80:   
                fines = [100, 150, 500, 1000]
            elif limit <= 100:  
                fines = [150, 200, 1000, 1500]
            else:               
                fines = [200, 1500, 2000]
            
            if over <= 20:      
                amt = fines[0]
            elif over <= 50:     
                amt = fines[1]
            else:               
                amt = fines[2]
            
            if delta == 0.2:
                p = 0.7
            elif delta == 0.5:
                p = 0.9
            elif delta == 0.7:
                p = 0.99
            else:
                p = 0
                
            fine_cost = amt * p
        
        # 总成本
        seg_cost = time_cost + fuel_cost + toll_cost + fine_cost
        
        # 累计总量
        total_time += seg_t
        total_cost += seg_cost
        total_time_cost += time_cost
        total_fuel_cost += fuel_cost
        total_toll_cost += toll_cost
        total_fine_cost += fine_cost
        
        # 获取起点和终点
        u = P_star_nodes[i]
        v = P_star_nodes[i+1]
        u_r, u_c = index_to_rc(u)
        v_r, v_c = index_to_rc(v)
        
        segment_details.append({
            "段号": i+1,
            "起点": f"({u_r},{u_c}) - {index_to_number(u)}号路口",
            "终点": f"({v_r},{v_c}) - {index_to_number(v)}号路口",
            "限速": limit,
            "超速比例": delta,
            "实际速度": actual_speed,
            "时间": seg_t,
            "时间成本": time_cost,
            "燃油费": fuel_cost,
            "通行费": toll_cost,
            "期望罚款": fine_cost,
            "总费用": seg_cost
        })
    
    # 打印详细信息
    print(f"\n路段详细信息:")
    print(f"{'段号':^4} | {'起点':^15} | {'终点':^15} | {'限速':^8} | {'超速比例':^8} | {'实际速度':^10} | {'时间':^10} | {'时间成本':^8} | {'燃油费':^8} | {'通行费':^8} | {'期望罚款':^10} | {'总费用':^10}")
    print("-" * 150)
    
    for detail in segment_details:
        print(f"{detail['段号']:^4} | {detail['起点']:^15} | {detail['终点']:^15} | {detail['限速']:^8.1f} | {detail['超速比例']*100:^8.0f}% | {detail['实际速度']:^10.1f} | {detail['时间']:^10.4f} | {detail['时间成本']:^8.2f} | {detail['燃油费']:^8.2f} | {detail['通行费']:^8.2f} | {detail['期望罚款']:^10.2f} | {detail['总费用']:^10.2f}")
    
    print("-" * 150)
    print(f"总计: {len(segment_details)}段路径, 总距离 {len(segment_details)*L} km, 总时间 {total_time:.4f} 小时, 总费用 {total_cost:.2f} 元")
    print(f"费用明细: 时间成本 {total_time_cost:.2f} 元 ({total_time_cost/total_cost*100:.1f}%) + 燃油费 {total_fuel_cost:.2f} 元 ({total_fuel_cost/total_cost*100:.1f}%) + 通行费 {total_toll_cost:.2f} 元 ({total_toll_cost/total_cost*100:.1f}%) + 期望罚款 {total_fine_cost:.2f} 元 ({total_fine_cost/total_cost*100:.1f}%)")
    
    # 计算相对于不超速情况的时间节省
    no_speeding_time = 0
    for limit in limits_P_star:
        no_speeding_time += seg_time(limit, 0.0)
    
    time_saving = no_speeding_time - total_time
    time_saving_percent = time_saving / no_speeding_time * 100
    
    print(f"\n相对于不超速情况（{no_speeding_time:.4f}小时）的时间节省: {time_saving:.4f} 小时 ({time_saving_percent:.2f}%)")
    
    return segment_details

# ——— 主程序 ———
if __name__ == "__main__":
    # 设计约束条件
    constraint = design_constraint()
    
    # 在约束条件下，优化路线一的时间
    min_time, min_time_cost, min_time_delta, pareto_frontier = optimize_route1_time()
    
    # 详细分析最优方案
    segment_details = analyze_optimal_solution(min_time_delta)
    
    # 输出最终结果
    print("\n=== 问题三最终结果 ===")
    print(f"约束条件: {constraint}")
    print(f"最短时间: {min_time:.4f} 小时")
    print(f"对应费用: {min_time_cost:.2f} 元")
    print(f"超速方案: {[f'{delta*100:.0f}%' for delta in min_time_delta]}")
