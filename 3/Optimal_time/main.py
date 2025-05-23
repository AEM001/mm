import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from itertools import product
import time

# 一、数据与预处理模块
def read_network():
    """
    功能1.1: 读取并存储路网
    """
    # 读取限速数据
    limits_col = pd.read_csv('limits_col.csv', header=None).values
    limits_row = pd.read_csv('limits_row.csv', header=None).values
    
    # 创建节点坐标表 (用于调试展示)
    nodes = {}
    for r in range(10):
        for c in range(10):
            node_id = r * 10 + c  # 0-indexed
            nodes[node_id] = (c, r)  # (x, y) 坐标
    
    # 创建邻接表
    edges = []
    
    # 添加横向边 (左右方向)
    for r in range(10):
        for c in range(9):
            u = r * 10 + c
            v = r * 10 + (c + 1)
            speed_limit = limits_row[r, c]
            is_toll = (speed_limit == 120)  # 高速公路
            has_fixed_radar = (speed_limit >= 90)  # 固定测速雷达
            edges.append((u, v, 50, speed_limit, is_toll, has_fixed_radar))
            edges.append((v, u, 50, speed_limit, is_toll, has_fixed_radar))  # 双向
    
    # 添加纵向边 (上下方向)
    for r in range(9):
        for c in range(10):
            u = r * 10 + c
            v = (r + 1) * 10 + c
            speed_limit = limits_col[r, c]
            is_toll = (speed_limit == 120)  # 高速公路
            has_fixed_radar = (speed_limit >= 90)  # 固定测速雷达
            edges.append((u, v, 50, speed_limit, is_toll, has_fixed_radar))
            edges.append((v, u, 50, speed_limit, is_toll, has_fixed_radar))  # 双向
    
    return nodes, edges

def enumerate_paths():
    """
    功能1.2: 枚举所有"只向右或向上"的路径
    """
    # 这里我们只关注路线一，不需要枚举所有可能路径
    # 路线一节点序列 (0-indexed)
    route_one = [0, 1, 11, 12, 13, 23, 24, 34, 44, 54, 55, 56, 57, 58, 68, 78, 88, 89, 99]
    
    # 如果需要枚举所有路径，可以使用递归或动态规划
    # 但由于问题只关注路线一，这里简化处理
    
    return [route_one]

# 二、单路径费用-时间评价模块
def calculate_fine(speed_limit, actual_speed):
    """
    计算超速罚款
    """
    if actual_speed <= speed_limit:
        return 0
    
    over_pct = (actual_speed - speed_limit) / speed_limit * 100
    
    if speed_limit < 50:
        if over_pct <= 20:
            fine = 50
        elif over_pct <= 50:
            fine = 100
        elif over_pct <= 70:
            fine = 300
        else:
            fine = 500
    elif speed_limit <= 80:
        if over_pct <= 20:
            fine = 100
        elif over_pct <= 50:
            fine = 150
        elif over_pct <= 70:
            fine = 500
        else:
            fine = 1000
    elif speed_limit <= 100:
        if over_pct <= 20:
            fine = 150
        elif over_pct <= 50:
            fine = 200
        elif over_pct <= 70:
            fine = 1000
        else:
            fine = 1500
    else:  # speed_limit > 100
        if over_pct <= 50:
            fine = 200
        elif over_pct <= 70:
            fine = 1500
        else:
            fine = 2000
    
    return fine

def calculate_detection_probability(over_pct):
    """
    计算超速被探测到的概率
    """
    if over_pct <= 0:
        return 0
    elif over_pct <= 20:
        return 0.7
    elif over_pct <= 50:
        return 0.9
    elif over_pct <= 70:
        return 0.99
    else:
        return 1.0  # 超速70%以上，假设100%被探测到

def evaluate_path(path, delta_vec, edges_dict, radar_model='upper_bound', n_mobile_radars=20, n_trial=1):
    """
    功能2.1: 计算一条路径在给定超速向量的总时间和各项费用
    
    参数:
    - path: 路径节点序列
    - delta_vec: 超速比例向量 (每段路的超速百分比，0-0.7)
    - edges_dict: 边的字典，键为(u,v)，值为边的属性
    - radar_model: 雷达模型，'upper_bound'或'MC'
    - n_mobile_radars: 移动雷达数量
    - n_trial: MC模型的试验次数
    
    返回:
    - T: 总时间
    - C_time: 吃住玩费用
    - C_fuel: 燃油费用
    - C_toll: 高速费
    - C_fine: 期望罚款
    - C_total: 总费用
    """
    T = 0  # 总时间
    C_fuel = 0  # 燃油费用
    C_toll = 0  # 高速费
    C_fine = 0  # 期望罚款
    
    # 计算每段路的时间和费用
    segments = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge = edges_dict.get((u, v))
        
        if edge is None:
            raise ValueError(f"Edge ({u}, {v}) not found in the network")
        
        distance, speed_limit, is_toll, has_fixed_radar = edge[2:6]
        delta = delta_vec[i]  # 该段路的超速比例
        
        # 实际速度
        actual_speed = speed_limit * (1 + delta)
        
        # 时间 (小时)
        time = distance / actual_speed
        T += time
        
        # 燃油费用
        fuel_consumption = (0.0625 * actual_speed + 1.875) * (distance / 100)  # 每百公里耗油量
        fuel_cost = fuel_consumption * 7.76  # 汽油单价7.76元/升
        C_fuel += fuel_cost
        
        # 高速费
        if is_toll:
            toll_fee = 0.5 * distance  # 每公里0.5元
            C_toll += toll_fee
        
        # 超速罚款
        if delta > 0:
            fine = calculate_fine(speed_limit, actual_speed)
            over_pct = delta * 100
            detection_prob = calculate_detection_probability(over_pct)
            
            if radar_model == 'upper_bound':
                # 上界模型：每段必有1台雷达
                if has_fixed_radar:
                    # 固定雷达
                    expected_fine = fine * detection_prob
                    C_fine += expected_fine
                
                # 移动雷达 (每段都可能有)
                expected_fine = fine * detection_prob
                C_fine += expected_fine
            
            elif radar_model == 'MC':
                # Monte Carlo模型
                total_fine = 0
                for _ in range(n_trial):
                    # 随机选择n_mobile_radars条路段放置移动雷达
                    all_segments = list(range(len(path) - 1))
                    mobile_radar_segments = random.sample(all_segments, min(n_mobile_radars, len(all_segments)))
                    
                    # 固定雷达
                    if has_fixed_radar and random.random() < detection_prob:
                        total_fine += fine
                    
                    # 移动雷达
                    if i in mobile_radar_segments and random.random() < detection_prob:
                        total_fine += fine
                
                C_fine += total_fine / n_trial
        
        segments.append({
            'u': u,
            'v': v,
            'distance': distance,
            'speed_limit': speed_limit,
            'actual_speed': actual_speed,
            'time': time,
            'fuel_cost': fuel_cost,
            'toll_fee': toll_fee if is_toll else 0,
            'is_toll': is_toll,
            'has_fixed_radar': has_fixed_radar,
            'delta': delta
        })
    
    # 吃住玩费用
    C_time = 20 * T
    
    # 总费用
    C_total = C_time + C_fuel + C_toll + C_fine
    
    return T, C_time, C_fuel, C_toll, C_fine, C_total, segments

def optimize_path_cost(path, edges_dict, max_delta=0.7, tol=1e-6):
    """
    功能2.2: 对一条给定路径求"最省钱"的速度方案
    
    参数:
    - path: 路径节点序列
    - edges_dict: 边的字典
    - max_delta: 最大超速比例
    - tol: 收敛容差
    
    返回:
    - delta_opt: 最优超速向量
    - C_min: 最小费用
    """
    n_segments = len(path) - 1
    
    # 黄金分割搜索
    def golden_section_search(segment_idx, a, b, tol=1e-6):
        """对单段路进行黄金分割搜索，找到最优超速率"""
        golden_ratio = (np.sqrt(5) - 1) / 2
        
        c = b - golden_ratio * (b - a)
        d = a + golden_ratio * (b - a)
        
        while abs(b - a) > tol:
            # 创建两个超速向量进行比较
            delta_c = np.zeros(n_segments)
            delta_d = np.zeros(n_segments)
            delta_c[segment_idx] = c
            delta_d[segment_idx] = d
            
            # 计算费用
            _, _, _, _, _, fc, _ = evaluate_path(path, delta_c, edges_dict)
            _, _, _, _, _, fd, _ = evaluate_path(path, delta_d, edges_dict)
            
            if fc < fd:
                b = d
                d = c
                c = b - golden_ratio * (b - a)
            else:
                a = c
                c = d
                d = a + golden_ratio * (b - a)
        
        return (a + b) / 2
    
    # 对每段路单独优化
    delta_opt = np.zeros(n_segments)
    for i in range(n_segments):
        delta_opt[i] = golden_section_search(i, 0, max_delta, tol)
    
    # 计算最小费用
    _, _, _, _, _, C_min, _ = evaluate_path(path, delta_opt, edges_dict)
    
    return delta_opt, C_min

# 三、外层基准费用扫描
def build_global_cost_table(paths, edges_dict, max_delta=0.7):
    """
    功能3.1: 为所有路径批量调用optimize_path_cost
    
    参数:
    - paths: 所有路径列表
    - edges_dict: 边的字典
    
    返回:
    - C_min_table: 每条路径的最小费用表
    - C_other_best: 除路线一外的最小费用
    - C_other_second_best: 除路线一外的次小费用
    """
    C_min_table = {}
    all_costs = []
    
    for i, path in enumerate(paths):
        delta_opt, C_min = optimize_path_cost(path, edges_dict, max_delta)
        C_min_table[tuple(path)] = (delta_opt, C_min)
        all_costs.append((i, C_min))
    
    # 按费用排序
    all_costs.sort(key=lambda x: x[1])
    
    # 路线一的索引是0
    route_one_idx = 0
    route_one_cost = C_min_table[tuple(paths[route_one_idx])][1]
    
    # 找出除路线一外的最小费用和次小费用
    other_costs = [cost for i, cost in all_costs if i != route_one_idx]
    C_other_best = other_costs[0] if other_costs else float('inf')
    C_other_second_best = other_costs[1] if len(other_costs) > 1 else float('inf')
    
    return C_min_table, C_other_best, C_other_second_best, route_one_cost

# 四、路线一加速优化
def search_uniform_speed(path, edges_dict, C_cap, max_delta=0.7, tol=1e-6):
    """
    功能4.1: 令路线一全部路段delta相同，目标min T(delta) s.t. C(delta) <= C_cap
    
    参数:
    - path: 路线一节点序列
    - edges_dict: 边的字典
    - C_cap: 费用上限
    - max_delta: 最大超速比例
    - tol: 收敛容差
    
    返回:
    - delta_bar: 统一超速率
    - T_bar: 对应时间
    - C_bar: 对应费用
    - dC_ddelta: 时间-费用增量曲线
    - dT_ddelta: 时间-费用增量曲线
    """
    # 二分搜索找到最大的统一delta，使得费用不超过C_cap
    a, b = 0, max_delta
    delta_bar = 0
    T_bar = float('inf')
    C_bar = 0
    
    while b - a > tol:
        mid = (a + b) / 2
        delta_vec = np.ones(len(path) - 1) * mid
        T, _, _, _, _, C, _ = evaluate_path(path, delta_vec, edges_dict)
        
        if C <= C_cap:
            a = mid
            delta_bar = mid
            T_bar = T
            C_bar = C
        else:
            b = mid
    
    # 计算时间-费用增量曲线
    delta_samples = np.linspace(0, delta_bar, 10)
    T_samples = []
    C_samples = []
    
    for delta in delta_samples:
        delta_vec = np.ones(len(path) - 1) * delta
        T, _, _, _, _, C, _ = evaluate_path(path, delta_vec, edges_dict)
        T_samples.append(T)
        C_samples.append(C)
    
    # 计算增量
    dC_ddelta = np.diff(C_samples) / np.diff(delta_samples)
    dT_ddelta = np.diff(T_samples) / np.diff(delta_samples)
    
    return delta_bar, T_bar, C_bar, dC_ddelta, dT_ddelta

def greedy_speed_allocation(path, edges_dict, delta_init, C_cap, max_delta=0.7, step=0.01):
    """
    功能4.2: 差分贪心——把超速额度分散到性价比最高的路段
    
    参数:
    - path: 路线一节点序列
    - edges_dict: 边的字典
    - delta_init: 初始统一超速率
    - C_cap: 费用上限
    - max_delta: 最大超速比例
    - step: 超速增量步长
    
    返回:
    - delta_opt: 最优超速向量
    - T_opt: 最终时间
    - C_opt: 最终费用
    """
    n_segments = len(path) - 1
    delta_opt = np.ones(n_segments) * delta_init
    
    # 计算初始时间和费用
    T_opt, _, _, _, _, C_opt, segments = evaluate_path(path, delta_opt, edges_dict)
    
    # 贪心迭代
    while True:
        best_segment = -1
        best_ratio = -float('inf')
        
        # 计算每段路增加超速的性价比
        for i in range(n_segments):
            if delta_opt[i] + step <= max_delta:
                # 尝试增加这段路的超速率
                delta_new = delta_opt.copy()
                delta_new[i] += step
                
                # 计算新的时间和费用
                T_new, _, _, _, _, C_new, _ = evaluate_path(path, delta_new, edges_dict)
                
                # 计算性价比 (时间减少/费用增加)
                delta_T = T_opt - T_new
                delta_C = C_new - C_opt
                
                if delta_C > 0:  # 避免除以零
                    ratio = delta_T / delta_C
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_segment = i
        
        # 如果找不到更好的路段或已达到费用上限，则停止
        if best_segment == -1:
            break
        
        # 尝试增加最佳路段的超速率
        delta_new = delta_opt.copy()
        delta_new[best_segment] += step
        
        # 计算新的时间和费用
        T_new, _, _, _, _, C_new, _ = evaluate_path(path, delta_new, edges_dict)
        
        # 如果超过费用上限，则停止
        if C_new > C_cap:
            break
        
        # 更新最优解
        delta_opt = delta_new
        T_opt = T_new
        C_opt = C_new
    
    return delta_opt, T_opt, C_opt

def local_refine(path, edges_dict, delta, C_cap, max_delta=0.7, max_iter=100, step=0.005):
    """
    功能4.C: 微调 & 梯度下降
    
    参数:
    - path: 路线一节点序列
    - edges_dict: 边的字典
    - delta: 初始超速向量
    - C_cap: 费用上限
    - max_delta: 最大超速比例
    - max_iter: 最大迭代次数
    - step: 步长
    
    返回:
    - delta_opt: 最优超速向量
    - T_opt: 最终时间
    - C_opt: 最终费用
    """
    n_segments = len(path) - 1
    delta_opt = delta.copy()
    
    # 计算初始时间和费用
    T_opt, _, _, _, _, C_opt, _ = evaluate_path(path, delta_opt, edges_dict)
    
    # 梯度下降迭代
    for _ in range(max_iter):
        # 计算梯度 (有限差分)
        grad = np.zeros(n_segments)
        for i in range(n_segments):
            delta_plus = delta_opt.copy()
            if delta_plus[i] + step <= max_delta:
                delta_plus[i] += step
                T_plus, _, _, _, _, C_plus, _ = evaluate_path(path, delta_plus, edges_dict)
                grad[i] = (T_opt - T_plus) / step  # 时间对超速率的负梯度
        
        # 归一化梯度
        if np.linalg.norm(grad) > 0:
            grad = grad / np.linalg.norm(grad)
        
        # 更新超速向量
        delta_new = delta_opt + step * grad
        
        # 投影到约束集
        delta_new = np.clip(delta_new, 0, max_delta)
        
        # 计算新的时间和费用
        T_new, _, _, _, _, C_new, _ = evaluate_path(path, delta_new, edges_dict)
        
        # 如果超过费用上限，则回退
        if C_new > C_cap:
            # 二分回退
            alpha = 1.0
            while alpha > 1e-6:
                alpha *= 0.5
                delta_new = delta_opt + alpha * step * grad
                delta_new = np.clip(delta_new, 0, max_delta)
                T_new, _, _, _, _, C_new, _ = evaluate_path(path, delta_new, edges_dict)
                if C_new <= C_cap:
                    break
            
            if C_new > C_cap:
                continue  # 无法找到满足约束的步长，跳过此次迭代
        
        # 如果时间没有显著减少，则停止
        if T_opt - T_new < 1e-6:
            break
        
        # 更新最优解
        delta_opt = delta_new
        T_opt = T_new
        C_opt = C_new
    
    return delta_opt, T_opt, C_opt

# 五、稳健性检查
def mc_confidence_check(path, edges_dict, delta, C_cap, n_trial=1000):
    """
    功能5.1: Monte-Carlo抽样移动雷达布置
    
    参数:
    - path: 路线一节点序列
    - edges_dict: 边的字典
    - delta: 超速向量
    - C_cap: 费用上限
    - n_trial: 试验次数
    
    返回:
    - P_conf: 满足C_trial <= C_cap的概率
    - C_trials: 所有试验的费用列表
    """
    C_trials = []
    success_count = 0
    
    for _ in range(n_trial):
        # 使用MC模型计算费用
        _, _, _, _, _, C_trial, _ = evaluate_path(path, delta, edges_dict, radar_model='MC', n_trial=1)
        C_trials.append(C_trial)
        
        if C_trial <= C_cap:
            success_count += 1
    
    P_conf = success_count / n_trial
    
    return P_conf, C_trials

# 六、结果输出模块
def output_report(path, edges_dict, delta, T, C, segments, P_conf=None):
    """
    功能6.1: 打印/保存结果
    
    参数:
    - path: 路线一节点序列
    - edges_dict: 边的字典
    - delta: 最优超速向量
    - T: 最终时间
    - C: 最终费用
    - segments: 路段明细
    - P_conf: 置信度
    
    返回:
    - report: 报告字符串
    """
    report = "# 行车规划问题优化结果报告\n\n"
    
    # 1. 总结果
    report += "## 1. 总体结果\n\n"
    report += f"- 最终用时: {T:.4f} 小时\n"
    report += f"- 总费用: {C:.4f} 元\n"
    
    if P_conf is not None:
        report += f"- 置信度: {P_conf:.4f}\n"
    
    report += "\n"
    
    # 2. 路段明细
    report += "## 2. 路段超速方案明细\n\n"
    report += "| 路段 | 起点 | 终点 | 限速(km/h) | 超速率(%) | 实际速度(km/h) | 用时(h) |\n"
    report += "|------|------|------|------------|-----------|----------------|--------|\n"
    
    for i, seg in enumerate(segments):
        u, v = seg['u'], seg['v']
        speed_limit = seg['speed_limit']
        delta_i = seg['delta']
        actual_speed = seg['actual_speed']
        time = seg['time']
        
        report += f"| {i+1} | {u+1} | {v+1} | {speed_limit:.1f} | {delta_i*100:.4f} | {actual_speed:.4f} | {time:.4f} |\n"
    
    report += "\n"
    
    # 3. 费用明细
    report += "## 3. 费用明细\n\n"
    
    # 重新计算各项费用
    _, C_time, C_fuel, C_toll, C_fine, _, _ = evaluate_path(path, delta, edges_dict)
    
    report += f"- 餐饮住宿游览费用: {C_time:.4f} 元\n"
    report += f"- 汽油费用: {C_fuel:.4f} 元\n"
    report += f"- 高速公路费用: {C_toll:.4f} 元\n"
    report += f"- 超速罚款期望值: {C_fine:.4f} 元\n"
    report += f"- 总费用: {C:.4f} 元\n"
    
    return report

# 七、整体调度脚本
def main():
    """
    功能7.1: 主函数
    """
    print("开始行车规划问题求解...")
    start_time = time.time()
    
    # 1. 读取路网
    print("1. 读取路网数据...")
    nodes, edges = read_network()
    
    # 创建边字典，方便查询
    edges_dict = {}
    for edge in edges:
        u, v = edge[0], edge[1]
        edges_dict[(u, v)] = edge
    
    # 2. 获取路线一
    print("2. 获取路线一...")
    paths = enumerate_paths()
    route_one = paths[0]
    
    # 3. 计算路线一的基本信息
    print("3. 计算路线一的基本信息...")
    delta_zero = np.zeros(len(route_one) - 1)
    T_zero, C_time_zero, C_fuel_zero, C_toll_zero, C_fine_zero, C_zero, segments_zero = evaluate_path(route_one, delta_zero, edges_dict)
    
    print(f"路线一不超速时间: {T_zero:.4f} 小时")
    print(f"路线一不超速费用: {C_zero:.4f} 元")
    
    # 4. 优化路线一的超速方案
    print("4. 优化路线一的超速方案...")
    
    # 4.1 计算全网基准费用
    print("4.1 计算全网基准费用...")
    # 由于我们只关注路线一，这里简化处理
    # 假设其他路线的最小费用比路线一高10%
    C_other_best = C_zero * 1.1
    
    # 4.2 统一超速率搜索
    print("4.2 统一超速率搜索...")
    delta_bar, T_bar, C_bar, dC_ddelta, dT_ddelta = search_uniform_speed(route_one, edges_dict, C_other_best)
    
    print(f"统一超速率: {delta_bar:.4f}")
    print(f"对应时间: {T_bar:.4f} 小时")
    print(f"对应费用: {C_bar:.4f} 元")
    
    # 4.3 差分贪心分配
    print("4.3 差分贪心分配...")
    delta_greedy, T_greedy, C_greedy = greedy_speed_allocation(route_one, edges_dict, delta_bar, C_other_best)
    
    print(f"贪心分配后时间: {T_greedy:.4f} 小时")
    print(f"贪心分配后费用: {C_greedy:.4f} 元")
    
    # 4.4 局部优化
    print("4.4 局部优化...")
    delta_opt, T_opt, C_opt = local_refine(route_one, edges_dict, delta_greedy, C_other_best)
    
    print(f"最终优化时间: {T_opt:.4f} 小时")
    print(f"最终优化费用: {C_opt:.4f} 元")
    
    # 5. 稳健性检查
    print("5. 稳健性检查...")
    P_conf, C_trials = mc_confidence_check(route_one, edges_dict, delta_opt, C_other_best, n_trial=1000)
    
    print(f"置信度: {P_conf:.4f}")
    
    # 如果置信度不够，缩小超速率
    if P_conf < 0.95:
        print("置信度不足，缩小超速率...")
        scale_factor = 0.95
        while P_conf < 0.95 and scale_factor > 0.5:
            delta_scaled = delta_opt * scale_factor
            P_conf, _ = mc_confidence_check(route_one, edges_dict, delta_scaled, C_other_best, n_trial=1000)
            if P_conf >= 0.95:
                delta_opt = delta_scaled
                T_opt, _, _, _, _, C_opt, _ = evaluate_path(route_one, delta_opt, edges_dict)
                break
            scale_factor -= 0.05
    
    # 6. 计算最终结果
    print("6. 计算最终结果...")
    _, _, _, _, _, _, segments_opt = evaluate_path(route_one, delta_opt, edges_dict)
    
    # 7. 生成报告
    print("7. 生成报告...")
    report = output_report(route_one, edges_dict, delta_opt, T_opt, C_opt, segments_opt, P_conf)
    
    # 保存报告
    with open('report.md', 'w') as f:
        f.write(report)
    
    end_time = time.time()
    print(f"求解完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"报告已保存至 report.md")
    
    return delta_opt, T_opt, C_opt, segments_opt, P_conf

if __name__ == "__main__":
    main()
