import numpy as np
import heapq
import matplotlib.pyplot as plt
from route1_extraction import *

# === 四）罚款函数与检测概率 ===

def calculate_fine(speed_limit, over_pct):
    """
    计算超速罚款金额
    
    参数:
        speed_limit: 限速（km/h）
        over_pct: 超速百分比（0.0-0.7）
        
    返回:
        罚款金额（元）
    """
    if over_pct <= 0:
        return 0
    
    # 将小数形式的百分比转换为整数百分比
    over_pct_int = over_pct * 100
    
    # 根据限速和超速百分比确定罚款金额
    if speed_limit < 50:
        # 低于50 km/h
        if over_pct_int <= 20:
            return 50
        elif over_pct_int <= 50:
            return 100
        elif over_pct_int <= 70:
            return 300
        else:
            return 500
    elif speed_limit <= 80:
        # 50-80 km/h
        if over_pct_int <= 20:
            return 100
        elif over_pct_int <= 50:
            return 150
        elif over_pct_int <= 70:
            return 500
        else:
            return 1000
    elif speed_limit <= 100:
        # 80-100 km/h
        if over_pct_int <= 20:
            return 150
        elif over_pct_int <= 50:
            return 200
        elif over_pct_int <= 70:
            return 1000
        else:
            return 1500
    else:
        # 高于100 km/h
        if over_pct_int <= 50:
            return 200
        elif over_pct_int <= 70:
            return 1500
        else:
            return 2000

def detection_probability(over_pct):
    """
    计算超速被任意一个雷达探测到的概率
    
    参数:
        over_pct: 超速百分比（0.0-0.7）
        
    返回:
        被探测到的概率（0.0-1.0）
    """
    if over_pct <= 0:
        return 0.0
    elif over_pct <= 0.2:
        return 0.7
    elif over_pct <= 0.5:
        return 0.9
    elif over_pct <= 0.7:
        return 0.99
    else:
        return 1.0  # 超过70%按100%概率处理

def expected_fine(speed_limit, over_pct):
    """
    计算期望罚款
    
    参数:
        speed_limit: 限速（km/h）
        over_pct: 超速百分比（0.0-0.7）
        
    返回:
        期望罚款金额（元）
    """
    if over_pct <= 0:
        return 0.0
    
    fine_amount = calculate_fine(speed_limit, over_pct)
    prob = detection_probability(over_pct)
    
    return fine_amount * prob

def segment_time(speed_limit, over_pct):
    """
    计算路段行驶时间
    
    参数:
        speed_limit: 限速（km/h）
        over_pct: 超速百分比（0.0-0.7）
        
    返回:
        行驶时间（小时）
    """
    actual_speed = speed_limit * (1 + over_pct)
    return L / actual_speed

def segment_cost(speed_limit, over_pct):
    """
    计算路段总成本（包括时间成本、燃油费、通行费和期望罚款）
    
    参数:
        speed_limit: 限速（km/h）
        over_pct: 超速百分比（0.0-0.7）
        
    返回:
        总成本（元）
    """
    # 实际速度
    actual_speed = speed_limit * (1 + over_pct)
    
    # 行驶时间（小时）
    travel_time = segment_time(speed_limit, over_pct)
    
    # 时间成本
    time_cost = time_cost_rate * travel_time
    
    # 燃油费
    fuel_consumption = (0.0625 * actual_speed + 1.875) * (L / 100.0)  # 油耗（升）
    fuel_cost = fuel_consumption * fuel_price
    
    # 高速通行费
    toll_cost = toll_per_km * L if speed_limit == 120 else 0.0
    
    # 期望罚款
    fine_cost = expected_fine(speed_limit, over_pct)
    
    # 总成本
    total_cost = time_cost + fuel_cost + toll_cost + fine_cost
    
    return total_cost

# 测试费用计算函数
def test_cost_functions():
    """测试费用计算函数"""
    print("\n=== 测试费用计算函数 ===")
    
    # 测试罚款计算
    print("\n1. 罚款金额测试:")
    test_cases = [
        (40, 0.1, "50元"),
        (40, 0.3, "100元"),
        (40, 0.6, "300元"),
        (60, 0.1, "100元"),
        (60, 0.3, "150元"),
        (60, 0.6, "500元"),
        (90, 0.1, "150元"),
        (90, 0.3, "200元"),
        (90, 0.6, "1000元"),
        (120, 0.1, "200元"),
        (120, 0.3, "200元"),
        (120, 0.6, "1500元")
    ]
    
    for speed_limit, over_pct, expected in test_cases:
        fine = calculate_fine(speed_limit, over_pct)
        print(f"限速{speed_limit}km/h, 超速{over_pct*100}%, 罚款: {fine}元 (预期: {expected})")
    
    # 测试检测概率
    print("\n2. 检测概率测试:")
    test_cases = [
        (0.0, "0%"),
        (0.1, "70%"),
        (0.2, "70%"),
        (0.3, "90%"),
        (0.5, "90%"),
        (0.6, "99%"),
        (0.7, "99%")
    ]
    
    for over_pct, expected in test_cases:
        prob = detection_probability(over_pct)
        print(f"超速{over_pct*100}%, 检测概率: {prob*100}% (预期: {expected})")
    
    # 测试期望罚款
    print("\n3. 期望罚款测试:")
    test_cases = [
        (60, 0.0, "0元"),
        (60, 0.2, "100元 * 70% = 70元"),
        (90, 0.5, "200元 * 90% = 180元"),
        (120, 0.7, "1500元 * 99% = 1485元")
    ]
    
    for speed_limit, over_pct, expected in test_cases:
        exp_fine = expected_fine(speed_limit, over_pct)
        print(f"限速{speed_limit}km/h, 超速{over_pct*100}%, 期望罚款: {exp_fine}元 (预期: {expected})")
    
    # 测试路段总成本
    print("\n4. 路段总成本测试:")
    test_cases = [
        (60, 0.0),
        (60, 0.2),
        (90, 0.5),
        (120, 0.7)
    ]
    
    for speed_limit, over_pct in test_cases:
        total_cost = segment_cost(speed_limit, over_pct)
        actual_speed = speed_limit * (1 + over_pct)
        travel_time = segment_time(speed_limit, over_pct)
        
        # 计算各项成本
        time_cost = time_cost_rate * travel_time
        fuel = (0.0625 * actual_speed + 1.875) * (L / 100.0)
        fuel_cost = fuel * fuel_price
        toll_cost = toll_per_km * L if speed_limit == 120 else 0.0
        fine_cost = expected_fine(speed_limit, over_pct)
        
        print(f"\n限速{speed_limit}km/h, 超速{over_pct*100}%, 实际速度{actual_speed}km/h:")
        print(f"  行驶时间: {travel_time:.4f}小时")
        print(f"  时间成本: {time_cost:.2f}元")
        print(f"  燃油费: {fuel_cost:.2f}元")
        print(f"  通行费: {toll_cost:.2f}元")
        print(f"  期望罚款: {fine_cost:.2f}元")
        print(f"  总成本: {total_cost:.2f}元")

# 运行测试
test_cost_functions()

# === 五）多标记Dijkstra算法 - 任意路径在给定时限内的最小费用 ===

def best_cost_any_path(T_lim, forbid_route1=False):
    """
    计算在给定时限内，任意路径的最小费用
    
    参数:
        T_lim: 时间限制（小时）
        forbid_route1: 是否禁止选择路线一
        
    返回:
        (最小费用, 是否是路线一, 最优路径, 超速方案)
    """
    print(f"计算时限{T_lim:.4f}小时内的最小费用路径...")
    
    # 每个节点维护一组"(time, cost)"的Pareto前沿状态
    labels = [[] for _ in range(100)]
    
    # 初始状态
    labels[0].append((0.0, 0.0, [], []))  # (time, cost, path, speed_plan)
    
    # 优先队列：(cost, time, node, path, speed_plan)
    pq = [(0.0, 0.0, 0, [], [])]
    
    # 记录最优解
    best_cost = float('inf')
    is_route1 = False
    best_path = None
    best_speed_plan = None
    
    # 记录处理过的状态数
    processed_states = 0
    
    while pq:
        cost, time, node, path, speed_plan = heapq.heappop(pq)
        processed_states += 1
        
        # 调试输出
        if processed_states % 10000 == 0:
            print(f"已处理{processed_states}个状态，当前队列长度：{len(pq)}")
        
        # 如果已经找到更好的解，跳过
        if cost > best_cost:
            continue
        
        # 如果到达终点，更新最优解
        if node == 99:  # 终点（100号路口）
            if time <= T_lim + 1e-9:
                # 检查是否是路线一
                current_is_route1 = (path == route1_indices)
                
                # 如果禁止路线一且当前路径是路线一，跳过
                if forbid_route1 and current_is_route1:
                    continue
                
                # 更新最优解
                if cost < best_cost - 1e-6:
                    best_cost = cost
                    is_route1 = current_is_route1
                    best_path = path.copy()
                    best_speed_plan = speed_plan.copy()
            continue
        
        # 遍历相邻路段
        for neighbor, limit in adj[node]:
            # 如果节点已经在当前路径中，跳过（避免环路）
            if neighbor in path:
                continue
                
            # 尝试不同的超速选项
            for delta in speed_options:
                # 计算该路段的成本和时间
                seg_time = segment_time(limit, delta)
                seg_cost = segment_cost(limit, delta)
                
                # 计算新的累计时间和成本
                new_time = time + seg_time
                new_cost = cost + seg_cost
                
                # 如果超出时间约束，跳过
                if new_time > T_lim + 1e-9:
                    continue
                
                # 更新路径和超速方案
                new_path = path + [neighbor]
                new_speed_plan = speed_plan + [delta]
                
                # 检查是否被Pareto支配
                dominated = False
                for t, c, _, _ in labels[neighbor]:
                    if t <= new_time + 1e-9 and c <= new_cost + 1e-9:
                        dominated = True
                        break
                
                if dominated:
                    continue
                
                # 移除被新状态支配的旧状态
                labels[neighbor] = [(t, c, p, s) for t, c, p, s in labels[neighbor] 
                                   if not (new_time <= t - 1e-9 and new_cost <= c - 1e-9)]
                
                # 添加新状态
                labels[neighbor].append((new_time, new_cost, new_path, new_speed_plan))
                
                # 加入优先队列
                heapq.heappush(pq, (new_cost, new_time, neighbor, new_path, new_speed_plan))
    
    print(f"算法结束，共处理{processed_states}个状态")
    
    # 如果没有找到可行解
    if best_path is None:
        print(f"未找到满足时间约束{T_lim:.4f}小时的可行路径！")
        return float('inf'), False, None, None
    
    return best_cost, is_route1, best_path, best_speed_plan

# === 六）二分法确定临界时限 ===

def find_critical_time_limit():
    """
    二分法确定临界时限
    
    返回:
        (临界时限, 第二名费用, 路线一在临界时限下的最小费用)
    """
    print("\n=== 二分法确定临界时限 ===")
    
    # 初始时限范围
    T_lo = 6.0   # 绝对下限（全段70%超速也跑不到再调小）
    T_hi = 12.0  # 宽松上限
    
    # 二分查找
    for iteration in range(20):  # 精度约0.02小时
        T_mid = 0.5 * (T_lo + T_hi)
        print(f"\n迭代 {iteration+1}: 测试时限 {T_mid:.4f} 小时")
        
        # 计算包含路线一的最小费用
        C_all, is_route1_optimal, _, _ = best_cost_any_path(T_mid)
        
        # 计算禁止路线一的最小费用（第二名）
        C_alt, _, _, _ = best_cost_any_path(T_mid, forbid_route1=True)
        
        print(f"全图最小费用: {C_all:.2f} 元, 是否是路线一: {is_route1_optimal}")
        print(f"第二名费用: {C_alt:.2f} 元")
        
        # 判断路线一是否已经是最便宜的
        if C_all <= C_alt + 1e-6 and is_route1_optimal:
            # 路线一已最便宜 → 可以压得更紧
            T_hi = T_mid
            print(f"路线一已是最便宜，可以压得更紧，更新上限: {T_hi:.4f}")
        else:
            # 还不是最便宜 → 时限还不够苛刻
            T_lo = T_mid
            print(f"路线一还不是最便宜，时限不够苛刻，更新下限: {T_lo:.4f}")
    
    # 最终临界时限
    T_crit = T_hi
    
    # 计算第二名费用
    C_alt, _, _, _ = best_cost_any_path(T_crit, forbid_route1=True)
    
    # 计算路线一在临界时限下的最小费用
    C_route1, _, _, _ = best_cost_any_path(T_crit)
    
    print(f"\n最终临界时限 T_crit = {T_crit:.4f} 小时")
    print(f"第二名费用 C_alt = {C_alt:.2f} 元")
    print(f"路线一在临界时限下的最小费用 = {C_route1:.2f} 元")
    
    return T_crit, C_alt, C_route1

# === 七）固定路线一，在费用约束下优化超速方案 ===

def optimize_route1_fixed_path(C_max=float('inf')):
    """
    固定路线一，在费用约束下优化超速方案
    
    参数:
        C_max: 费用约束（元）
        
    返回:
        (最小时间, 对应费用, 最优超速方案)
    """
    print(f"\n优化路线一的超速方案（费用上限: {C_max:.2f} 元）")
    
    # 初始状态: (time, cost, speed_plan)
    state = [(0.0, 0.0, [])]
    
    # 对路线一的每段路进行动态规划
    for i, limit in enumerate(route1_limits):
        print(f"处理第{i+1}段路，限速{limit} km/h...")
        
        # 下一个状态集合
        next_state = []
        
        # 对当前所有状态，尝试所有超速选项
        for time, cost, speed_plan in state:
            for delta in speed_options:
                # 计算该路段的时间和成本
                seg_time = segment_time(limit, delta)
                seg_cost = segment_cost(limit, delta)
                
                # 计算新的累计时间和成本
                new_time = time + seg_time
                new_cost = cost + seg_cost
                
                # 检查费用约束
                if new_cost <= C_max + 1e-9:
                    next_state.append((new_time, new_cost, speed_plan + [delta]))
        
        # Pareto剪枝 - 保留时间最短的方案
        next_state.sort(key=lambda x: (x[0], x[1]))  # 按时间和费用排序
        
        # 保留Pareto前沿
        pareto_frontier = []
        best_cost = float('inf')
        
        for new_time, new_cost, new_speed_plan in next_state:
            if new_cost < best_cost - 1e-9:
                pareto_frontier.append((new_time, new_cost, new_speed_plan))
                best_cost = new_cost
        
        state = pareto_frontier
        print(f"当前Pareto前沿大小: {len(state)}")
    
    # 如果没有找到可行解
    if not state:
        print("未找到满足费用约束的可行方案！")
        return float('inf'), float('inf'), None
    
    # 选择时间最短的方案
    min_time_solution = min(state, key=lambda x: x[0])
    min_time, min_time_cost, min_time_speed_plan = min_time_solution
    
    print(f"\n最优方案: 时间 = {min_time:.4f} 小时, 费用 = {min_time_cost:.2f} 元")
    print(f"超速方案: {[f'{delta*100:.0f}%' for delta in min_time_speed_plan]}")
    
    return min_time, min_time_cost, min_time_speed_plan

# === 八）问题三主函数 ===

def solve_problem3():
    """
    解决问题三
    
    返回:
        (最短时间, 对应费用, 最优超速方案)
    """
    print("\n=== 解决问题三 ===")
    
    # 1. 二分法确定临界时限和第二名费用
    T_crit, C_alt, C_route1 = find_critical_time_limit()
    
    # 2. 在第二名费用约束下，优化路线一的超速方案
    min_time, min_time_cost, optimal_speed_plan = optimize_route1_fixed_path(C_max=C_alt)
    
    print(f"\n问题三最终结果:")
    print(f"临界时限: {T_crit:.4f} 小时")
    print(f"第二名费用: {C_alt:.2f} 元")
    print(f"最短时间: {min_time:.4f} 小时")
    print(f"对应费用: {min_time_cost:.2f} 元")
    print(f"超速方案: {[f'{delta*100:.0f}%' for delta in optimal_speed_plan]}")
    
    return min_time, min_time_cost, optimal_speed_plan

# 运行问题三求解
min_time, total_cost, optimal_speed_plan = solve_problem3()

# === 九）结果可视化 ===

def visualize_optimal_speed_plan(speed_plan, title):
    """
    可视化最优超速方案
    
    参数:
        speed_plan: 超速方案
        title: 图表标题
    """
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
    for node_idx in route1_indices:
        r, c = index_to_rc(node_idx)
        # 转换为坐标系（从0开始）
        y = r - 1
        x = c - 1
        path_x.append(x)
        path_y.append(y)
    
    # 使用颜色表示超速程度
    cmap = plt.colormaps.get_cmap('RdYlGn_r')  # 红黄绿色谱，反转使红色表示高超速
    
    for i in range(len(route1_indices) - 1):
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
    for i, node_idx in enumerate(route1_indices):
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

# 可视化最优超速方案
if optimal_sp
(Content truncated due to size limit. Use line ranges to read in chunks)