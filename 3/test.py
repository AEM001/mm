import heapq
import json
import math
import random

# --- 常量定义 ---
FUEL_PRICE = 7.76  # 元/升
SEGMENT_DISTANCE = 50.0  # km
NUM_MOBILE_RADARS = 20
TOTAL_SEGMENTS_IN_GRID = 180  # 10x10 网格: (10 条线 * 9 段/线) * 2 方向 (对于双向路段的计数方式)
PROB_MOBILE_RADAR_EFFECTIVELY_ON_SEGMENT = NUM_MOBILE_RADARS / TOTAL_SEGMENTS_IN_GRID # 大约 1/9

# 超速选项: key -> {factor: 速度倍数, over_percentage: 超速百分比, detection_prob_single_radar: 单个雷达探测概率}
SPEEDING_OPTIONS = {
    0.0: {'factor': 1.0, 'over_percentage': 0.0, 'detection_prob_single_radar': 0.0}, # 不超速
    0.2: {'factor': 1.2, 'over_percentage': 0.2, 'detection_prob_single_radar': 0.7}, # 超速20%
    0.5: {'factor': 1.5, 'over_percentage': 0.5, 'detection_prob_single_radar': 0.9}, # 超速50%
    0.7: {'factor': 1.7, 'over_percentage': 0.7, 'detection_prob_single_radar': 0.99},# 超速70%
}

# DP中成本的缩放因子 (处理小数点后两位)
COST_SCALE_FACTOR = 100

# --- 网格和图的辅助函数 ---
def get_node_id(x, y):
    """将 (x,y) 坐标 (x,y 均为 0-9) 转换为节点 ID (1-100)。(0,0) 是左下角。"""
    if not (0 <= x <= 9 and 0 <= y <= 9):
        return None
    return y * 10 + x + 1

def get_node_coords(node_id):
    """将节点 ID (1-100) 转换为 (x,y) 坐标。"""
    if not (1 <= node_id <= 100):
        return None, None
    return (node_id - 1) % 10, (node_id - 1) // 10

def parse_csv_string_to_list(csv_string):
    """将CSV格式的字符串解析为整数列表的列表。"""
    lines = csv_string.strip().split('\n')
    data = []
    for line in lines:
        data.append([int(val) for val in line.split(',')])
    return data

def load_actual_speed_limits(limits_col_str, limits_row_str):
    """
    从CSV字符串加载实际的路段限速数据。
    limits_col_str: 纵向限速的CSV字符串 (9x10)
    limits_row_str: 横向限速的CSV字符串 (10x9)
    返回: {(u,v): speed_limit} 的字典
    """
    segment_speeds = {}
    limits_col_data = parse_csv_string_to_list(limits_col_str)
    limits_row_data = parse_csv_string_to_list(limits_row_str)

    num_rows_col = 9
    num_cols_col = 10
    for r_idx in range(num_rows_col): 
        for c_idx in range(num_cols_col): 
            start_node_x, start_node_y = c_idx, r_idx
            end_node_x, end_node_y = c_idx, r_idx + 1
            u = get_node_id(start_node_x, start_node_y)
            v = get_node_id(end_node_x, end_node_y)
            speed = limits_col_data[r_idx][c_idx]
            if u is not None and v is not None:
                segment_speeds[(u, v)] = speed
                segment_speeds[(v, u)] = speed 

    num_rows_row = 10
    num_cols_row = 9
    for r_idx in range(num_rows_row): 
        for c_idx in range(num_cols_row): 
            start_node_x, start_node_y = c_idx, r_idx
            end_node_x, end_node_y = c_idx + 1, r_idx
            u = get_node_id(start_node_x, start_node_y)
            v = get_node_id(end_node_x, end_node_y)
            speed = limits_row_data[r_idx][c_idx]
            if u is not None and v is not None:
                segment_speeds[(u, v)] = speed
                segment_speeds[(v, u)] = speed 
                
    print(f"从提供的数据加载了 {len(segment_speeds)} 个路段限速条目 (双向计算)。")
    return segment_speeds

# --- 成本计算函数 ---
def get_fine_amount(speed_limit, actual_speed):
    """根据限速和实际速度计算罚款金额。"""
    if actual_speed <= speed_limit or speed_limit <= 0:
        return 0.0
    
    over_percentage = (actual_speed - speed_limit) / speed_limit

    if speed_limit < 50:
        if over_percentage <= 0.2: return 50.0
        if over_percentage <= 0.5: return 100.0
        if over_percentage <= 0.7: return 300.0
        return 500.0
    elif speed_limit <= 80:
        if over_percentage <= 0.2: return 100.0
        if over_percentage <= 0.5: return 150.0
        if over_percentage <= 0.7: return 500.0
        return 1000.0
    elif speed_limit <= 100:
        if over_percentage <= 0.2: return 150.0
        if over_percentage <= 0.5: return 200.0
        if over_percentage <= 0.7: return 1000.0
        return 1500.0
    else: 
        if over_percentage <= 0.5: return 200.0
        if over_percentage <= 0.7: return 1500.0
        return 2000.0

def calculate_segment_details(base_speed_limit, speeding_option_key):
    """计算路段的时间、油费、活动费、过路费和期望罚款。"""
    option = SPEEDING_OPTIONS[speeding_option_key]
    actual_speed = base_speed_limit * option['factor']
    
    if actual_speed <= 1e-6: # 避免实际速度过小或为零导致除零错误
        time_hours = float('inf')
        fuel_cost = float('inf')
        activity_cost = float('inf')
    else:
        time_hours = SEGMENT_DISTANCE / actual_speed
        liters_per_100km = 0.0625 * actual_speed + 1.875
        fuel_cost = (liters_per_100km / 100.0) * SEGMENT_DISTANCE * FUEL_PRICE
        activity_cost = 20 * time_hours

    is_highway = (base_speed_limit == 120) 
    toll_cost = 0.5 * SEGMENT_DISTANCE if is_highway else 0.0

    fine_if_caught = get_fine_amount(base_speed_limit, actual_speed)
    prob_detection_by_single_radar = option['detection_prob_single_radar']
    
    has_fixed_radar = (base_speed_limit >= 90)
    prob_not_detected_by_fixed = (1.0 - prob_detection_by_single_radar) if (has_fixed_radar and speeding_option_key > 0.0) else 1.0
    prob_not_detected_by_mobile_setup = (1.0 - PROB_MOBILE_RADAR_EFFECTIVELY_ON_SEGMENT) + \
                                       (PROB_MOBILE_RADAR_EFFECTIVELY_ON_SEGMENT * ((1.0 - prob_detection_by_single_radar) if speeding_option_key > 0.0 else 1.0) )
    overall_prob_not_detected = prob_not_detected_by_fixed * prob_not_detected_by_mobile_setup
    prob_detected = (1.0 - overall_prob_not_detected) if speeding_option_key > 0.0 else 0.0
    expected_fine = fine_if_caught * prob_detected

    if time_hours == float('inf'): # 如果时间是无限的，成本也应该是无限的
        total_expected_cost_float = float('inf')
    else:
        total_expected_cost_float = fuel_cost + activity_cost + toll_cost + expected_fine
    
    total_expected_cost_scaled = int(round(total_expected_cost_float * COST_SCALE_FACTOR)) if total_expected_cost_float != float('inf') else float('inf')


    return {
        'time': time_hours,
        'cost_float': total_expected_cost_float,
        'cost_scaled': total_expected_cost_scaled,
        'actual_speed': actual_speed,
        'details': { 
            'fuel': fuel_cost, 'activity': activity_cost, 'toll': toll_cost, 
            'fine_amount_if_caught': fine_if_caught, 'prob_overall_detect': prob_detected, 'expected_fine': expected_fine,
            'base_limit': base_speed_limit, 'speeding_factor': option['factor']
        }
    }

def precompute_all_segment_options(segment_speed_limits_map):
    """为所有路段和所有超速选项预计算 (时间, 浮点成本, 缩放后成本)。"""
    all_options = {}
    for (u, v), speed_limit in segment_speed_limits_map.items():
        if (u,v) not in all_options: all_options[(u,v)] = {}
        for skey in SPEEDING_OPTIONS:
            all_options[(u,v)][skey] = calculate_segment_details(speed_limit, skey)
    return all_options

# --- Dijkstra 算法 ---
def dijkstra(graph_nodes, start_node, end_node, all_segment_options_data, weight_selector_func):
    """
    使用 Dijkstra 算法查找最短路径。
    返回: (总权重, 路径节点列表, 路径选项列表 (如果适用)) 或 (None, [], None)
    """
    dist = {node: float('inf') for node in graph_nodes}
    prev = {node: None for node in graph_nodes}
    
    dist[start_node] = 0
    pq = [(0, start_node)] 

    adj = {node: [] for node in graph_nodes}
    for (u_adj,v_adj) in all_segment_options_data.keys():
        if u_adj in graph_nodes and v_adj in graph_nodes:
            adj[u_adj].append(v_adj)

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        if u == end_node: break 
        
        for v in adj.get(u, []): 
            segment_data_uv = all_segment_options_data.get((u,v), None)
            if not segment_data_uv: continue

            weight, _ = weight_selector_func(u, v, segment_data_uv) 

            if weight == float('inf'): continue

            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
    
    path = []
    curr = end_node
    if dist[curr] == float('inf'): return None, [], None 

    while curr is not None:
        path.append(curr)
        curr = prev[curr]
    path.reverse()

    if not path or path[0] != start_node : return None, [], None

    final_path_options = None
    if weight_selector_func.__name__ == 'c_bound_weight_selector': 
        final_path_options = []
        for i in range(len(path) - 1):
            u_seg, v_seg = path[i], path[i+1]
            _ , key_for_segment = weight_selector_func(u_seg, v_seg, all_segment_options_data[(u_seg, v_seg)])
            final_path_options.append(key_for_segment)

    return dist[end_node], path, final_path_options


# --- 主要逻辑 ---
def solve_problem():
    print("开始解决问题...")
    start_junction_id = 1
    end_junction_id = 100
    num_junctions_side = 10 
    all_junction_ids = list(range(1, num_junctions_side * num_junctions_side + 1))

    limits_col_csv_data = """
60,90,60,60,40,40,40,60,60,120
40,60,60,60,40,90,60,60,60,120
40,60,60,60,120,90,60,40,40,120
40,60,90,60,120,60,60,60,40,120
60,60,90,60,120,60,60,60,40,40
60,40,90,60,120,60,40,40,40,40
40,40,90,40,120,40,40,40,90,40
60,60,90,40,40,60,60,60,90,60
60,60,90,60,60,60,90,60,60,90
"""
    limits_row_csv_data = """
90,40,40,40,40,40,40,40,40
60,90,90,40,40,60,60,40,40
40,60,60,60,60,60,40,40,40
40,60,90,90,90,60,60,60,60
60,60,60,60,60,60,60,60,40
60,60,90,90,90,90,90,90,60
40,40,60,60,40,40,40,60,40
60,60,40,40,40,40,90,90,90
120,120,120,120,40,40,90,90,90
60,60,60,60,40,60,60,60,60
"""
    segment_speed_limits_map = load_actual_speed_limits(limits_col_csv_data, limits_row_csv_data)
    
    if not segment_speed_limits_map:
        print("错误：未能加载路段限速数据。")
        return

    print("正在为所有选项预计算路段成本和时间...")
    all_seg_opts_data = precompute_all_segment_options(segment_speed_limits_map)
    
    print("正在确定路线一 (最短时间，不超速)...")
    def route_one_weight_selector(u, v, segment_options):
        return segment_options[0.0]['time'], 0.0 
    
    min_time_no_speeding, route_one_nodes, _ = dijkstra(
        all_junction_ids, start_junction_id, end_junction_id, 
        all_seg_opts_data, route_one_weight_selector
    )

    if not route_one_nodes or (len(route_one_nodes)>0 and route_one_nodes[0] != start_junction_id) :
        print("错误：未能找到路线一 (初始最短时间路径)。")
        return
    
    print(f"路线一 (不超速) 已找到: {route_one_nodes}")
    print(f"路线一 (不超速) 的时间: {min_time_no_speeding:.4f} 小时")

    route_one_segments = []
    for i in range(len(route_one_nodes) - 1):
        route_one_segments.append((route_one_nodes[i], route_one_nodes[i+1]))

    # 计算路线一在不超速情况下的总成本
    cost_route_one_no_speeding_float = 0.0
    for seg_uv_r1 in route_one_segments:
        if seg_uv_r1 in all_seg_opts_data and 0.0 in all_seg_opts_data[seg_uv_r1]:
            cost_route_one_no_speeding_float += all_seg_opts_data[seg_uv_r1][0.0]['cost_float']
        else:
            print(f"错误：路线一的片段 {seg_uv_r1} 或其不超速选项在预计算数据中缺失。")
            cost_route_one_no_speeding_float = float('inf') # 标记为错误状态
            break
    if cost_route_one_no_speeding_float != float('inf'):
        print(f"路线一 (不超速) 的期望总成本: {cost_route_one_no_speeding_float:.4f} 元")


    print("正在计算 C_bound (任何路线可达到的最低期望成本)...")
    def c_bound_weight_selector(u, v, segment_options):
        min_cost = float('inf')
        best_key = None
        for skey_option, data_option in segment_options.items():
            if data_option['cost_float'] < min_cost:
                min_cost = data_option['cost_float']
                best_key = skey_option
        return min_cost, best_key

    c_bound_float, c_bound_path_nodes, c_bound_path_options = dijkstra(
        all_junction_ids, start_junction_id, end_junction_id,
        all_seg_opts_data, c_bound_weight_selector
    )

    if c_bound_float is None or c_bound_float == float('inf'): 
        print("错误：未能计算有效的 C_bound (可能所有路径成本都为无穷大)。")
        return
    
    W_capacity_scaled = int(round(c_bound_float * COST_SCALE_FACTOR)) 
    print(f"C_bound (浮点数): {c_bound_float:.4f}, DP的缩放后容量 W: {W_capacity_scaled}")

    print(f"正在为路线一 (共 {len(route_one_segments)} 段) 求解 MCKP...")
    dp_prev = [float('inf')] * (W_capacity_scaled + 1)
    dp_prev[0] = 0.0 

    path_choices_trace = [{} for _ in range(len(route_one_segments))]

    for seg_idx, segment_uv in enumerate(route_one_segments):
        dp_curr = [float('inf')] * (W_capacity_scaled + 1) 
        current_segment_options = all_seg_opts_data[segment_uv]
        
        for prev_c_scaled in range(W_capacity_scaled + 1): 
            if dp_prev[prev_c_scaled] == float('inf'):
                continue 

            for skey, option_data in current_segment_options.items():
                if option_data['cost_scaled'] == float('inf') or option_data['time'] == float('inf'):
                    continue

                seg_cost_scaled = option_data['cost_scaled']
                seg_time = option_data['time']
                
                new_total_c_scaled = prev_c_scaled + seg_cost_scaled
                
                if new_total_c_scaled <= W_capacity_scaled:
                    new_total_time = dp_prev[prev_c_scaled] + seg_time
                    if new_total_time < dp_curr[new_total_c_scaled]: 
                        dp_curr[new_total_c_scaled] = new_total_time
                        path_choices_trace[seg_idx][new_total_c_scaled] = (skey, prev_c_scaled)
        dp_prev = dp_curr 

    final_dp_table = dp_prev
    final_min_time = float('inf')
    final_scaled_cost_achieved = -1

    for c_scaled in range(W_capacity_scaled + 1):
        if final_dp_table[c_scaled] < final_min_time:
            final_min_time = final_dp_table[c_scaled]
            final_scaled_cost_achieved = c_scaled

    # --- 处理DP结果 ---
    if final_min_time == float('inf'):
        print(f"\n在 C_bound ({c_bound_float:.4f} 元) 约束下，未能为路线一找到任何可行的超速方案。")
        if cost_route_one_no_speeding_float != float('inf'): # 确保不超速成本有效
            print(f"路线一在不超速情况下的成本为 {cost_route_one_no_speeding_float:.4f} 元，时间为 {min_time_no_speeding:.4f} 小时。")
            if cost_route_one_no_speeding_float <= c_bound_float:
                print("这通常意味着任何旨在减少时间的超速行为都会导致路线一的成本超过 C_bound。")
                print("因此，在保持成本效益（成本 <= C_bound）的前提下，时间无法进一步降低。")
                print("\n--- 问题三结果 (基于无法在C_bound内优化时间) ---")
                print(f"路线一 (原始路径): 节点序列 {route_one_nodes}")
                print(f"  路线一包含的路段数量: {len(route_one_segments)}")
                print(f"\n由于无法在C_bound内通过超速减少时间，最优方案为不超速:")
                print(f"  最短时间: {min_time_no_speeding:.4f} 小时 (未降低)")
                print(f"  对应期望总成本: {cost_route_one_no_speeding_float:.4f} 元")
                print("\n  路线一各路段超速方案:")
                for i, seg_uv_r1_print in enumerate(route_one_segments):
                    print(f"  路段 {i+1}: {seg_uv_r1_print} -> 超速因子 x_k = {0.0}")
            else: # cost_route_one_no_speeding_float > c_bound_float
                print("路线一在不超速的情况下，其成本已经高于 C_bound。")
                print("因此，在题目“费用最优的路线仍然保持为路线一”的严格约束（成本 <= C_bound）下，无法找到符合条件的方案。")
        else:
            print("由于计算路线一不超速成本时出错，无法提供进一步分析。")
        return # DP未找到解，结束

    # --- 如果DP找到解 ---
    speeding_plan_route_one = [None] * len(route_one_segments)
    current_c_scaled_for_trace = final_scaled_cost_achieved
    
    for seg_idx in range(len(route_one_segments) - 1, -1, -1):
        if current_c_scaled_for_trace in path_choices_trace[seg_idx]:
            skey, prev_c_scaled_from_trace = path_choices_trace[seg_idx][current_c_scaled_for_trace]
            speeding_plan_route_one[seg_idx] = skey
            current_c_scaled_for_trace = prev_c_scaled_from_trace
        else:
            print(f"错误：回溯时在路段 {seg_idx} (目标成本 {current_c_scaled_for_trace}) 出现问题。")
            print(f"  Trace for segment {seg_idx}: {path_choices_trace[seg_idx].keys() if path_choices_trace and len(path_choices_trace) > seg_idx else 'Trace not available'}")
            for k_idx in range(seg_idx, -1, -1): 
                if speeding_plan_route_one[k_idx] is None: 
                     speeding_plan_route_one[k_idx] = 0.0
            break 

    recalculated_final_cost_float = 0.0
    recalculated_final_time = 0.0
    valid_plan = True
    for i, seg_uv in enumerate(route_one_segments):
        chosen_skey = speeding_plan_route_one[i]
        if chosen_skey is None: 
            chosen_skey = 0.0 
            speeding_plan_route_one[i] = 0.0 
            valid_plan = False

        if seg_uv not in all_seg_opts_data or chosen_skey not in all_seg_opts_data[seg_uv]:
            print(f"错误：路段 {seg_uv} 或超速选项 {chosen_skey} 的数据在预计算中缺失。")
            valid_plan = False; recalculated_final_cost_float = float('inf'); recalculated_final_time = float('inf'); break 
            
        details = all_seg_opts_data[seg_uv][chosen_skey]
        if details['cost_float'] == float('inf') or details['time'] == float('inf'):
            print(f"警告：路段 {i+1} {seg_uv} 选择的方案 ({chosen_skey}) 成本或时间为无限。")
            valid_plan = False; recalculated_final_cost_float = float('inf'); recalculated_final_time = float('inf'); break

        recalculated_final_cost_float += details['cost_float']
        recalculated_final_time += details['time']

    print("\n--- 问题三结果 ---")
    print(f"路线一 (原始路径): 节点序列 {route_one_nodes}")
    print(f"  路线一包含的路段数量: {len(route_one_segments)}")
    
    print(f"\n路线一在超速情况下的最优方案 (总期望成本 ≤ C_bound={c_bound_float:.4f}):")
    print(f"  目标最短时间 (来自DP): {final_min_time:.4f} 小时")
    print(f"  对应期望总成本 (来自DP, 浮点数): {final_scaled_cost_achieved / COST_SCALE_FACTOR:.4f} 元")
    
    if valid_plan and recalculated_final_time != float('inf'):
        print(f"\n  根据选择的超速方案重新计算:")
        print(f"  重新计算得到的最短时间: {recalculated_final_time:.4f} 小时")
        print(f"  重新计算得到的期望总成本: {recalculated_final_cost_float:.4f} 元")

    print("\n  路线一各路段超速方案 (路段索引: (u, v) -> 超速因子 x_k):")
    for i, seg_uv_print in enumerate(route_one_segments):
        print(f"  路段 {i+1}: {seg_uv_print} -> 超速因子 x_k = {speeding_plan_route_one[i]}")

    if final_min_time < min_time_no_speeding:
        print(f"\n时间成功从 {min_time_no_speeding:.4f} 小时降低到 {final_min_time:.4f} 小时。")
    elif final_min_time != float('inf'): # 如果找到了方案，但时间没有减少
        print(f"\n时间未能降低或与原时间相同。原始时间: {min_time_no_speeding:.4f}, 新时间: {final_min_time:.4f}。")
    # 如果 final_min_time 是 inf，则之前已经打印过相关信息

    if valid_plan and final_min_time != float('inf') and \
       (abs(recalculated_final_time - final_min_time) > 1e-3 or \
        abs(recalculated_final_cost_float - (final_scaled_cost_achieved / COST_SCALE_FACTOR)) > 1e-3) :
        print("\n警告：DP结果与根据方案重新计算的值之间存在差异。请检查DP逻辑或回溯过程。")
    elif not valid_plan and final_min_time != float('inf'): # DP找到了解，但回溯或重计算出问题
        print("\n警告：由于回溯或数据问题，最终方案的验证可能不准确或不完整。")


if __name__ == '__main__':
    solve_problem()

