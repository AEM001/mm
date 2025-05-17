import pandas as pd
import numpy as np
import networkx as nx
import pulp as pl
from tabulate import tabulate

# 读取限速矩阵
def load_speed_limits():
    row_limits = pd.read_csv('limits_row.csv', header=None).values  # 横向限速 (10×9)
    col_limits = pd.read_csv('limits_col.csv', header=None).values  # 纵向限速 (9×10)
    return row_limits, col_limits

# 构建有向图
def build_graph(row_limits, col_limits):
    G = nx.DiGraph()
    d = 50  # 每段路长 50 km
    for r in range(10):
        for c in range(10):
            idx = r*10 + c
            # 向右
            if c < 9:
                v = row_limits[r, c]
                toll = 50 * 0.5 if v == 120 else 0  # 高速公路 0.5元/km ⇒ 25元/段
                G.add_edge(idx, idx+1, d=d, v_lim=v, toll=toll)
                G.add_edge(idx+1, idx, d=d, v_lim=v, toll=toll)
            # 向上
            if r < 9:
                v = col_limits[r, c]
                toll = 50 * 0.5 if v == 120 else 0
                G.add_edge(idx, idx+10, d=d, v_lim=v, toll=toll)
                G.add_edge(idx+10, idx, d=d, v_lim=v, toll=toll)
    return G

# 提取路线一的路段和限速
def extract_route_one(G):
    # 路线一的节点序列 (0-indexed)
    route_one_nodes = [0, 1, 11, 12, 13, 23, 24, 34, 44, 54, 55, 56, 57, 58, 68, 78, 88, 89, 99]
    
    # 提取路段和限速
    route_one_segments = []
    for i in range(len(route_one_nodes) - 1):
        from_node = route_one_nodes[i]
        to_node = route_one_nodes[i+1]
        edge_data = G.get_edge_data(from_node, to_node)
        
        segment = {
            'edge_id': i,
            'from_node': from_node,
            'to_node': to_node,
            'v_lim': edge_data['v_lim'],
            'd': edge_data['d'],
            'toll': edge_data['toll']
        }
        route_one_segments.append(segment)
    
    return route_one_segments

# 计算罚款
def calculate_fine(speed_limit, over_pct):
    # 添加一个小的epsilon值避免浮点误差
    epsilon = 1e-6
    
    if over_pct <= 0:
        return 0
    elif speed_limit < 50:
        if over_pct <= 20 + epsilon:
            return 50
        elif over_pct <= 50 + epsilon:
            return 100
        elif over_pct <= 70 + epsilon:
            return 300
        else:
            return 500
    elif speed_limit <= 80:
        if over_pct <= 20 + epsilon:
            return 100
        elif over_pct <= 50 + epsilon:
            return 150
        elif over_pct <= 70 + epsilon:
            return 500
        else:
            return 1000
    elif speed_limit <= 100:
        if over_pct <= 20 + epsilon:
            return 150
        elif over_pct <= 50 + epsilon:
            return 200
        elif over_pct <= 70 + epsilon:
            return 1000
        else:
            return 1500
    else:  # speed_limit > 100
        if over_pct <= 50 + epsilon:
            return 200
        elif over_pct <= 70 + epsilon:
            return 1500
        else:
            return 2000

# 计算探测概率 - 直接使用题目给定的整段概率
def detection_probability(over_pct):
    # 添加一个小的epsilon值避免浮点误差
    epsilon = 1e-6
    
    if over_pct <= 0:
        return 0
    elif over_pct <= 20 + epsilon:
        return 0.70
    elif over_pct <= 50 + epsilon:
        return 0.90
    elif over_pct <= 70 + epsilon:
        return 0.99
    else:
        return 1.0  # 超过70%一定会被探测到

# 计算期望罚款 - 修正为直接使用整段概率
def expected_penalty(speed_limit, over_pct):
    fine = calculate_fine(speed_limit, over_pct)
    prob = detection_probability(over_pct)
    return fine * prob

# 计算油费
def calculate_fuel_cost(distance, speed, over_pct):
    actual_speed = speed * (1 + over_pct/100)
    fuel_consumption = (0.0625 * actual_speed + 1.875) * (distance / 100)  # 每百公里耗油量
    return fuel_consumption * 7.76  # 汽油单价7.76元/升

# 计算行驶时间
def calculate_travel_time(distance, speed, over_pct):
    actual_speed = speed * (1 + over_pct/100)
    return distance / actual_speed  # 小时

# 计算餐饮住宿费
def calculate_meal_cost(travel_time):
    return 20 * travel_time  # 20元/小时

# 计算总费用
def calculate_total_cost(segment, over_pct):
    v_lim = segment['v_lim']
    distance = segment['d']
    toll = segment['toll']
    
    travel_time = calculate_travel_time(distance, v_lim, over_pct)
    meal_cost = calculate_meal_cost(travel_time)
    fuel_cost = calculate_fuel_cost(distance, v_lim, over_pct)
    penalty = expected_penalty(v_lim, over_pct)
    
    return {
        'travel_time': travel_time,
        'meal_cost': meal_cost,
        'fuel_cost': fuel_cost,
        'toll': toll,
        'penalty': penalty,
        'total_cost': meal_cost + fuel_cost + toll + penalty
    }

# 设计超速方案 - 添加费用约束
def design_speeding_scheme(route_segments):
    # 定义可能的超速率（百分比）- 上限设为69%而非70%
    speeding_rates = [0, 5, 10, 15, 19, 35, 49, 60, 69]
    
    # 计算基准路线费用（不超速情况）
    base_cost = sum(calculate_total_cost(segment, 0)['total_cost'] for segment in route_segments)
    print(f"基准路线费用（不超速）: {base_cost:.4f} 元")
    
    # 创建优化模型
    model = pl.LpProblem("Route_Optimization", pl.LpMinimize)
    
    # 为每个路段的每个可能超速率创建二元变量
    x = {}
    for i, segment in enumerate(route_segments):
        for k, rate in enumerate(speeding_rates):
            x[i, k] = pl.LpVariable(f"x_{i}_{k}", cat=pl.LpBinary)
    
    # 每个路段只能选择一个超速率
    for i in range(len(route_segments)):
        model += pl.lpSum(x[i, k] for k in range(len(speeding_rates))) == 1
    
    # 预计算每个路段在每个超速率下的时间和费用
    time_table = {}
    cost_table = {}
    for i, segment in enumerate(route_segments):
        for k, rate in enumerate(speeding_rates):
            result = calculate_total_cost(segment, rate)
            time_table[i, k] = result['travel_time']
            cost_table[i, k] = result['total_cost']
    
    # 添加总费用约束 - 确保不超过基准路线费用
    total_cost_expr = pl.lpSum(cost_table[i, k] * x[i, k] for i in range(len(route_segments)) 
                              for k in range(len(speeding_rates)))
    model += total_cost_expr <= base_cost + 1e-6  # 添加一个小的epsilon值避免浮点误差
    
    # 目标函数：最小化总行驶时间
    model += pl.lpSum(time_table[i, k] * x[i, k] for i in range(len(route_segments)) 
                     for k in range(len(speeding_rates)))
    
    # 求解模型
    model.solve(pl.PULP_CBC_CMD(msg=False))
    
    # 提取结果
    results = []
    total_time = 0
    total_cost = 0
    
    for i, segment in enumerate(route_segments):
        selected_k = None
        for k in range(len(speeding_rates)):
            if pl.value(x[i, k]) == 1:
                selected_k = k
                break
        
        selected_rate = speeding_rates[selected_k]
        result = calculate_total_cost(segment, selected_rate)
        
        total_time += result['travel_time']
        total_cost += result['total_cost']
        
        results.append({
            'edge_id': i,
            'from_node': segment['from_node'],
            'to_node': segment['to_node'],
            'v_lim': segment['v_lim'],
            'speeding_rate': selected_rate,
            'travel_time': result['travel_time'],
            'meal_cost': result['meal_cost'],
            'fuel_cost': result['fuel_cost'],
            'toll': result['toll'],
            'penalty': result['penalty'],
            'total_cost': result['total_cost']
        })
    
    return results, total_time, total_cost, base_cost

# 主函数
def main():
    # 加载限速数据
    row_limits, col_limits = load_speed_limits()
    
    # 构建图
    G = build_graph(row_limits, col_limits)
    
    # 提取路线一的路段
    route_one_segments = extract_route_one(G)
    
    # 设计超速方案
    results, total_time, total_cost, base_cost = design_speeding_scheme(route_one_segments)
    
    # 输出结果
    print(f"最短时间: {total_time:.4f} 小时")
    print(f"总费用: {total_cost:.4f} 元")
    print(f"时间减少: {(base_time - total_time):.4f} 小时 ({(base_time - total_time) / base_time * 100:.2f}%)")
    
    # 创建表格
    table_data = []
    for result in results:
        table_data.append([
            result['edge_id'],
            f"{result['from_node']}→{result['to_node']}",
            result['v_lim'],
            f"{result['speeding_rate']:.2f}",
            f"{result['travel_time']:.4f}",
            f"{result['total_cost']:.4f}"
        ])
    
    headers = ["Edge ID", "from→to", "v_lim", "s (超速率%)", "τ_e (h)", "ΔC_e (¥)"]
    print(tabulate(table_data, headers=headers, tablefmt="pipe"))
    
    # 保存结果到文件
    with open('results_corrected.md', 'w') as f:
        f.write("# 行车规划问题：超速方案优化结果（修正版）\n\n")
        f.write(f"## 基准路线费用（不超速）: {base_cost:.4f} 元\n")
        f.write(f"## 最短时间: {total_time:.4f} 小时\n")
        f.write(f"## 总费用: {total_cost:.4f} 元\n")
        f.write(f"## 时间减少: {(base_time - total_time):.4f} 小时 ({(base_time - total_time) / base_time * 100:.2f}%)\n\n")
        f.write("## 每个路段的超速方案\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt="pipe"))
        f.write("\n\n")
        f.write("| **汇总** |  |  |  | " + f"**T* = {total_time:.4f} h** | **C* = {total_cost:.4f} ¥** |")
        
        # 添加详细的费用明细
        f.write("\n\n## 详细费用明细\n\n")
        detail_headers = ["Edge ID", "from→to", "v_lim", "s (%)", "时间(h)", "餐饮费(¥)", "油费(¥)", "通行费(¥)", "罚款(¥)", "总费用(¥)"]
        detail_data = []
        for result in results:
            detail_data.append([
                result['edge_id'],
                f"{result['from_node']}→{result['to_node']}",
                result['v_lim'],
                f"{result['speeding_rate']:.2f}",
                f"{result['travel_time']:.4f}",
                f"{result['meal_cost']:.4f}",
                f"{result['fuel_cost']:.4f}",
                f"{result['toll']:.4f}",
                f"{result['penalty']:.4f}",
                f"{result['total_cost']:.4f}"
            ])
        f.write(tabulate(detail_data, headers=detail_headers, tablefmt="pipe"))

# 计算基准时间（不超速情况）
def calculate_base_time(route_segments):
    return sum(calculate_total_cost(segment, 0)['travel_time'] for segment in route_segments)

if __name__ == "__main__":
    # 加载限速数据
    row_limits, col_limits = load_speed_limits()
    
    # 构建图
    G = build_graph(row_limits, col_limits)
    
    # 提取路线一的路段
    route_one_segments = extract_route_one(G)
    
    # 计算基准时间
    base_time = calculate_base_time(route_one_segments)
    print(f"基准时间（不超速）: {base_time:.4f} 小时")
    
    # 运行主函数
    main()
