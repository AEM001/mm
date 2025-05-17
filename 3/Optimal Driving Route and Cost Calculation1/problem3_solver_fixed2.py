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

def expected_fine(speed_limit, over_pct, has_radar=True):
    """
    计算期望罚款
    
    参数:
        speed_limit: 限速（km/h）
        over_pct: 超速百分比（0.0-0.7）
        has_radar: 是否有固定雷达（默认为True，简化计算）
        
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

def segment_cost(speed_limit, over_pct, has_radar=True):
    """
    计算路段总成本（包括时间成本、燃油费、通行费和期望罚款）
    
    参数:
        speed_limit: 限速（km/h）
        over_pct: 超速百分比（0.0-0.7）
        has_radar: 是否有固定雷达（默认为True，简化计算）
        
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
    fine_cost = expected_fine(speed_limit, over_pct, has_radar)
    
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

# === 五）计算路线一在不同超速方案下的时间和费用 ===

def calculate_route1_all_speeding_options():
    """
    计算路线一在所有可能的超速方案下的时间和费用
    
    返回:
        所有可能的超速方案及其时间和费用
    """
    print("\n=== 计算路线一在所有超速方案下的时间和费用 ===")
    
    # 计算不超速情况下的总时间和费用
    total_time_no_speeding = 0
    total_cost_no_speeding = 0
    
    for limit in route1_limits:
        total_time_no_speeding += segment_time(limit, 0.0)
        total_cost_no_speeding += segment_cost(limit, 0.0)
    
    print(f"不超速情况下: 总时间 = {total_time_no_speeding:.4f} 小时, 总费用 = {total_cost_no_speeding:.2f} 元")
    
    # 计算全部最大超速情况下的总时间和费用
    total_time_max_speeding = 0
    total_cost_max_speeding = 0
    
    for limit in route1_limits:
        total_time_max_speeding += segment_time(limit, 0.7)
        total_cost_max_speeding += segment_cost(limit, 0.7)
    
    print(f"全部最大超速情况下: 总时间 = {total_time_max_speeding:.4f} 小时, 总费用 = {total_cost_max_speeding:.2f} 元")
    
    # 使用链式DP优化超速方案
    print("\n使用链式DP优化超速方案...")
    
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
    
    # 找出时间最短的方案
    min_time_solution = min(state, key=lambda x: x[0])
    min_time, min_time_cost, min_time_speed_plan = min_time_solution
    
    print(f"\n时间最短方案: 时间 = {min_time:.4f} 小时, 费用 = {min_time_cost:.2f} 元")
    print(f"超速方案: {[f'{delta*100:.0f}%' for delta in min_time_speed_plan]}")
    
    # 找出费用最低的方案
    min_cost_solution = min(state, key=lambda x: x[1])
    min_cost_time, min_cost, min_cost_speed_plan = min_cost_solution
    
    print(f"\n费用最低方案: 时间 = {min_cost_time:.4f} 小时, 费用 = {min_cost:.2f} 元")
    print(f"超速方案: {[f'{delta*100:.0f}%' for delta in min_cost_speed_plan]}")
    
    # 计算所有方案的时间和费用
    all_solutions = []
    
    for time, cost, speed_plan in state:
        all_solutions.append((time, cost, speed_plan))
    
    # 按时间排序
    all_solutions.sort(key=lambda x: x[0])
    
    print(f"\n所有Pareto最优方案（按时间排序）:")
    for i, (time, cost, _) in enumerate(all_solutions):
        print(f"方案 {i+1}: 时间 = {time:.4f} 小时, 费用 = {cost:.2f} 元")
    
    return all_solutions, min_time, min_time_cost, min_time_speed_plan

# 计算路线一在所有超速方案下的时间和费用
all_solutions, min_time, min_time_cost, optimal_speed_plan = calculate_route1_all_speeding_options()

# === 六）计算其他路径在给定时限内的最小费用 ===

def best_cost_other_paths(T_lim):
    """
    计算除路线一外的其他路径在给定时限内的最小费用
    
    参数:
        T_lim: 时间限制（小时）
        
    返回:
        (最小费用, 最优路径, 超速方案)
    """
    print(f"\n=== 计算除路线一外的其他路径在时限{T_lim:.4f}小时内的最小费用 ===")
    
    # 每个节点维护一组"(time, cost)"的Pareto前沿状态
    labels = [[] for _ in range(100)]
    
    # 初始状态
    labels[0].append((0.0, 0.0, [], []))  # (time, cost, path, speed_plan)
    
    # 优先队列：(cost, time, node, path, speed_plan)
    pq = [(0.0, 0.0, 0, [], [])]
    
    # 记录最优解
    best_cost = float('inf')
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
                
                # 如果是路线一，跳过
                if current_is_route1:
                    continue
                
                # 更新最优解
                if cost < best_cost - 1e-6:
                    best_cost = cost
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
        return float('inf'), None, None
    
    # 转换为原图编号
    best_path_numbers = [index_to_number(idx) for idx in best_path]
    
    print(f"最小费用: {best_cost:.2f} 元")
    print(f"最优路径: {best_path_numbers}")
    
    return best_cost, best_path, best_speed_plan

# === 七）二分法确定临界时限 ===

def find_critical_time_limit():
    """
    二分法确定临界时限
    
    返回:
        (临界时限, 第二名费用)
    """
    print("\n=== 二分法确定临界时限 ===")
    
    # 获取路线一的所有可能方案
    route1_solutions = all_solutions
    
    # 按时间排序
    route1_solutions.sort(key=lambda x: x[0])
    
    # 初始时限范围
    T_lo = route1_solutions[0][0]  # 路线一的最短时间
    T_hi = route1_solutions[-1][0]  # 路线一的最长时间
    
    print(f"初始时限范围: [{T_lo:.4f}, {T_hi:.4f}]")
    
    # 二分查找
    for iteration in range(20):  # 精度约0.02小时
        T_mid = 0.5 * (T_lo + T_hi)
        print(f"\n迭代 {iteration+1}: 测试时限 {T_mid:.4f} 小时")
        
        # 找到路线一在T_mid时限下的最小费用
        route1_min_cost = float('inf')
        for time, cost, _ in route1_solutions:
            if time <= T_mid + 1e-9 and cost < route1_min_cost:
                route1_min_cost = cost
        
        # 计算其他路径在T_mid时限下的最小费用
        other_min_cost, _, _ = best_cost_other_paths(T_mid)
        
        print(f"路线一最小费用: {route1_min_cost:.2f} 元")
        print(f"其他路径最小费用: {other_min_cost:.2f} 元")
        
        # 判断路线一是否已经是最便宜的
        if route1_min_cost <= other_min_cost + 1e-6:
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
    other_min_cost, _, _ = best_cost_other_paths(T_crit)
    
    # 计算路线一在临界时限下的最小费用
    route1_min_cost = float('inf')
    for time, cost, _ in route1_solutions:
        if time <= T_crit + 1e-9 and cost < route1_min_cost:
            route1_min_cost = cost
    
    print(f"\n最终临界时限 T_crit = {T_crit:.4f} 小时")
    print(f"路线一在临界时限下的最小费用 = {route1_min_cost:.2f} 元")
    print(f"第二名费用 = {other_min_cost:.2f} 元")
    
    return T_crit, other_min_cost

# === 八）在第二名费用约束下，找到路线一的最短时间方案 ===

def find_min_time_under_cost_constraint(C_max):
    """
    在费用约束下，找到路线一的最短时间方案
    
    参数:
        C_max: 费用约束（元）
        
    返回:
        (最短时间, 对应费用, 最优超速方案)
    """
    print(f"\n=== 在费用约束{C_max:.2f}元下，找到路线一的最短时间方案 ===")
    
    # 找到满足费用约束的最短时间方案
    min_time = float('inf')
    min_time_cost = float('inf')
    min_time_speed_plan = None
    
    for time, cost, speed_plan in all_solutions:
        if cost <= C_max + 1e-9 and time < min_time:
            min_time = time
            min_time_cost = cost
            min_time_speed_plan = speed_plan
    
    if min_time_speed_plan is None:
        print(f"未找到满足费用约束{C_max:.2f}元的可行方案！")
        return float('inf'), float('inf'), None
    
    print(f"最短时间: {min_time:.4f} 小时")
    print(f"对应费用: {min_time_cost:.2f} 元")
    print(f"超速方案: {[f'{delta*100:.0f}%' for delta in min_time_speed_plan]}")
    
    return min_time, min_time_cost, min_time_speed_plan

# === 九）问题三主函数 ===

def solve_problem3():
    """
    解决问题三
    
    返回:
        (最短时间, 对应费用, 最优超速方案)
    """
    print("\n=== 解决问题三 ===")
    
    # 1. 二分法确定临界时限和第二名费用
    T_crit, C_alt = find_critical_time_limit()
    
    # 2. 在第二名费用约束下，找到路线一的最短时间方案
    min_time, min_time_cost, optimal_speed_plan = find_min_time_under_cost_constraint(C_alt)
    
    print(f"\n问题三最终结果:")
    print(f"临界时限: {T_crit:.4f} 小时")
    print(f"第二名费用: {C_alt:.2f} 元")
    print(f"最短时间: {min_time:.4f} 小时")
    print(f"对应费用: {min_time_cost:.2f} 元")
    print(f"超速方案: {[f'{delta*100:.0f}%' for delta in optimal_speed_plan]}")
    
    return min_time, min_time_cost, optimal_speed_plan

# 运行问题三求解
min_time, total_cost, optimal_speed_plan = solve_problem3()

# === 十）结果可视化 ===

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
if optimal_speed_plan is not None:
    visualize_optimal_speed_plan(optimal_speed_plan, "问题三最优超速方案(修正两步法)")
    print("\n已生成最优超速方案可视化图: 问题三最优超速方案(修正两步法).png")

# === 十一）详细分析最优方案 ===

def analyze_optimal_solution(speed_plan):
    """
    详细分析最优超速方案
    
    参数:
        speed_plan: 超速方案
    """
    if speed_plan is None:
        print("\n无法分析最优方案，因为未找到满足约束的可行解")
        return None
        
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
        limit = route1_limits[i]
        actual_speed = limit * (1 + delta)
        
        # 计算时间和成本
        seg_time = segment_time(limit, delta)
        time_cost = time_cost_rate * seg_time
        
        # 燃油费
        fuel_consumption = (0.0625 * actual_speed + 1.875) * (L / 100.0)
        fuel_cost = fuel_consumption * fuel_price
        
        # 高速通行费
        toll_cost = toll_per_km * L if limit == 120 else 0.0
        
        # 期望罚款
        fine_cost = expected_fine(limit, delta)
        
        # 总成本
        seg_cost = time_cost + fuel_cost + toll_cost + fine_cost
        
        # 累计总量
        total_time += seg_time
        total_cost += seg_cost
        total_time_cost += time_cost
        total_fuel_cost += fuel_cost
        total_toll_cost += toll_cost
        total_fine_cost += fine_cost
        
        # 获取起点和终点
        u = route1_indices[i]
        v = route1_indices[i+1]
        u_r, u_c = index_to_rc(u)
        v_r, v_c = index_to_rc(v)
        
        segment_details.append({
            "段号": i+1,
            "起点": f"({u_r},{u_c}) - {index_to_number(u)}号路口",
            "终点": f"({v_r},{v_c}) - {index_to_number(v)}号路口",
            "限速": limit,
            "超速比例": delta,
            "实际速度": actual_speed,
            "时间": seg_time,
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
    print(f"费用明细: 时间成本 {total_time_cost:.2f} 元 + 燃油费 {total_fuel_cost:.2f} 元 + 通行费 {total_toll_cost:.2f} 元 + 期望罚款 {total_fine_cost:.2f} 元")
    
    # 计算相对于问题一不超速情况的时间节省
    time_saving = total_time_no_speeding - total_time
    time_saving_percent = time_saving / total_time_no_speeding * 100
    
    print(f"\n相对于问题一不超速情况（{total_time_no_speeding:.4f}小时）的时间节省: {time_saving:.4f} 小时 ({time_saving_percent:.2f}%)")
    
    return segment_details

# 分析最优方案
segment_details = analyze_optimal_solution(optimal_speed_plan)
