import numpy as np
import pandas as pd
from heapq import heappush, heappop

# 0. 用户提供的 T1 值 - 将被动态计算取代
# T1 = 10.8333  # hours

# 1. 数据准备
# 1.1 读入 CSV (路径调整到 data 文件夹下)
# 使用原始字符串 (r'') 来避免转义序列警告
limits_col = pd.read_csv(r'..\data\limits_col.csv', header=None).values  # 9×10
limits_row = pd.read_csv(r'..\data\limits_row.csv', header=None).values  # 10×9

# 1.2 构造 180 条有向边
EDGE_LEN = 50  # km
edges = []     # [from_id, to_id, v_max]

# 竖向（向上行驶 和 向下行驶）——节点编号：r*10+c 与 (r+1)*10+c 之间
for r in range(9):
    for c in range(10):
        n1, n2 = r*10+c+1, (r+1)*10+c+1      # +1 因题目节点从 1 开始
        v_limit = limits_col[r, c]
        edges.append((n1, n2, v_limit))
        edges.append((n2, n1, v_limit)) # 添加反向边

# 横向（向右行驶 和 向左行驶）
for r in range(10):
    for c in range(9):
        n1, n2 = r*10+c+1, r*10+(c+1)+1
        v_limit = limits_row[r, c]
        edges.append((n1, n2, v_limit))
        edges.append((n2, n1, v_limit)) # 添加反向边

# 2. 预计算 4 档速度的 时间 & 成本表
BRACKETS = [0.0, 0.2, 0.5, 0.7]  # 超速倍率
GAS_PRICE = 7.76                 # 元/升
C_FOOD = 20                      # 元/小时
TOLL_RATE = 0.5                  # 高速每公里
EDGE_KM = 50                     # 每段路程长度与 EDGE_LEN 一致

def fuel_consumption(v_speed):  # 每 100 km 耗油量
    return 0.0625*v_speed + 1.875  # L/100km

def fine_amount(v_max_limit, x_ratio):
    ratio = x_ratio
    v_class = v_max_limit
    # 分段法规（摘录自题干）
    # 注意：以下边界条件 (如 v_class < 50, v_class <= 80) 严格按照题目文字描述。
    # 如有官方示例或判例与此不同，需按官方为准。
    if v_class < 50:
        if ratio <= .2:  return 50
        if ratio <= .5:  return 100
        if ratio <= .7:  return 300
        return 500
    elif v_class <= 80:
        if ratio <= .2:  return 100
        if ratio <= .5:  return 150
        if ratio <= .7:  return 500
        return 1000
    elif v_class <= 100:
        if ratio <= .2:  return 150
        if ratio <= .5:  return 200
        if ratio <= .7:  return 1000
        return 1500
    else:  # >100
        if ratio <= .5:  return 200
        if ratio <= .7:  return 1500
        return 2000

# 固定雷达：v_max ≥ 90 的边必有
fixed_radar = lambda v_max_limit: v_max_limit >= 90
MOBILE_NUM = 20
# EDGE_NUM 会在边列表构建完成后更新
# rho 会在 EDGE_NUM 更新后重新计算

label_cost = {}  # (edge_id, bracket_idx) -> (time_h, cost)

# 更新 EDGE_NUM 和 rho 在边列表完全构建之后
EDGE_NUM = len(edges)
rho = MOBILE_NUM / EDGE_NUM      # 移动雷达到达某条边的概率（平均化假设）


for eid, (u_node, v_node, vmax_limit) in enumerate(edges):
    # 有无雷达
    has_fix_radar = fixed_radar(vmax_limit) # Renamed to avoid conflict with p_fix
    for k, x_val in enumerate(BRACKETS):
        v_real = vmax_limit * (1 + x_val)
        if v_real == 0: # Avoid division by zero if speed is somehow zero
            t_hr = float('inf')
        else:
            t_hr = EDGE_KM / v_real

        # ——油费
        gas_cost = fuel_consumption(v_real) * EDGE_KM/100 * GAS_PRICE if v_real > 0 else 0
        # ——餐饮 / 住宿 / 游览
        travel_cost = C_FOOD * t_hr if t_hr != float('inf') else float('inf')
        # ——高速收费
        toll_cost = (EDGE_KM * TOLL_RATE) if vmax_limit == 120 else 0

        # ——期望罚款
        P_det = 0
        if x_val > 0:  # 只在超速时才有罚款
            p_single = 0.7 if x_val <= 0.2 else 0.9 if x_val <= 0.5 else 0.99
            
            # 根据建议修正 P_det 计算
            # P_det = 1 - (1-p)^{\text{fix}} (1-\rho p).
            # (1-p)^fix term: (1-p_single) if has_fix_radar else 1
            term_fix = (1 - p_single) if has_fix_radar else 1.0
            # E[(1-p)^N_mob] = 1 - rho * p_single (This is an approximation, more accurately (1-rho*p_single))
            # A more standard interpretation for "probability rho of encountering a mobile radar on an edge"
            # P(detected by mobile) = rho * p_single
            # P(not detected by mobile) = 1 - rho * p_single
            # P(not detected by fixed) = (1 - p_single) if has_fix_radar else 1.0
            # P(not detected at all) = P(not detected by fixed) * P(not detected by mobile if fixed didn't get you)
            # Assuming independence: P(not detected) = P_not_fixed * P_not_mobile
            # P_not_fixed = (1-p_single) if has_fix_radar else 1.0
            # P_not_mobile_given_edge = 1 - (rho * p_single) # Prob of not being caught by mobile radar if one is on this edge and you speed
            # P_det = 1 - P_not_detected
            # P_not_detected = P_not_detected_by_fixed * P_not_detected_by_mobile
            # P_not_detected_by_fixed = (1-p_single) if has_fix_radar else 1.0
            # P_not_detected_by_mobile = (1 - rho * p_single) # This assumes rho is the probability a mobile radar IS on this specific edge AND it detects you.
                                                            # Or, if rho is average density, P(mobile radar on edge) = rho. P(detected by it) = p_single.
                                                            # P(detected by at least one) = 1 - P(detected by none)
                                                            # P(not detected by fixed) = 1 if no fixed radar, (1-p_single) if fixed radar
                                                            # P(not detected by mobile) = 1 - rho*p_single (if rho is prob of mobile radar being on THIS edge)
            
            # Let's stick to the formula provided in comments if it's from a reliable source for the problem context
            # P_det = 1.0 - term_fix * (1 - rho * p_single) # Original interpretation from user's comment
            # A common alternative: P_det = P(fix) + P(mob and not fix) = P(fix) + P(mob) * (1-P(fix))
            # If P(fix) = p_single if has_fix_radar else 0
            # If P(mob) = rho * p_single
            # P_fix_effective = p_single if has_fix_radar else 0.0
            # P_mob_effective = rho * p_single
            # P_det = 1 - (1 - P_fix_effective) * (1 - P_mob_effective)

            p_detect_fixed = p_single if has_fix_radar else 0.0
            p_detect_mobile = rho * p_single # Probability of being detected by a mobile radar on this segment

            P_det = 1 - (1 - p_detect_fixed) * (1 - p_detect_mobile)


        exp_fine = P_det * fine_amount(vmax_limit, x_val) if x_val > 0 else 0

        total = gas_cost + travel_cost + toll_cost + exp_fine
        if t_hr == float('inf'): # If time is infinite, cost should also be infinite
            total = float('inf')
        label_cost[(eid, k)] = (t_hr, total)

start, goal = 1, 100

# 新增：dijkstra_time_only 函数，用于计算 T1
def dijkstra_time_only(graph_edges_input, num_nodes, start_node, end_node, edge_len_km_input):
    dist_time = [float('inf')] * (num_nodes + 1)
    dist_time[start_node] = 0
    pq_time = [(0, start_node)] # (time, node)

    # 构建邻接表 (仅包含时间和目标节点)
    adj_time = [[] for _ in range(num_nodes + 1)]
    # graph_edges_input is expected to be like `edges`: list of (u, v, vmax)
    for u_orig, v_orig, vmax_orig in graph_edges_input: 
        if vmax_orig == 0: continue # 不可通行
        time_taken_local = edge_len_km_input / vmax_orig
        adj_time[u_orig].append((v_orig, time_taken_local))
        # Assuming graph_edges_input already contains bidirectional edges if needed,
        # or this function is called with a graph that is inherently directed for this purpose.
        # The original `edges` list IS bidirectional.

    while pq_time:
        d, u = heappop(pq_time)

        if d > dist_time[u]:
            continue
        if u == end_node:
            return dist_time[end_node]

        for v_neighbor, time_to_neighbor in adj_time[u]:
            if dist_time[u] + time_to_neighbor < dist_time[v_neighbor]:
                dist_time[v_neighbor] = dist_time[u] + time_to_neighbor
                heappush(pq_time, (dist_time[v_neighbor], v_neighbor))
    return float('inf') # 如果无法到达终点

# 计算 T1
# `edges` 列表已经是 (from, to, vmax) 并且是双向的
print("Calculating T1 (baseline time)...")
T1 = dijkstra_time_only(edges, 100, start, goal, EDGE_KM)
if T1 == float('inf'):
    print("Error: Could not calculate T1. Goal is unreachable under normal speed limits.")
    # exit() # 或者设置一个默认T1并警告
    # For now, let's set a placeholder if T1 is inf to avoid crashes, but this indicates a problem.
    T1 = 24.0 # Placeholder T1 if unreachable, ideally handle this more gracefully
    print(f"Warning: Goal unreachable for T1 calculation. Using placeholder T1 = {T1} hours.")
else:
    print(f"Calculated T1 = {T1:.4f} hours.")

# 3. 离散化时间轴
T_max = 0.7 * T1
SLOT = 0.5/60      # 0.5 分钟 (30 秒)
N_T = int(np.ceil(T_max / SLOT))
if N_T <= 0:
    print(f"Error: N_T is {N_T}. T_max ({T_max:.4f}h) might be too small or SLOT ({SLOT*60:.1f}min) too large.")
    N_T = 1 # Ensure N_T is at least 1 to avoid empty dist arrays if proceeding
    print(f"Adjusted N_T to {N_T}.")


# 4. 构造扩展图 Label-Setting 最短路
INF = float('inf') # Use float('inf') for consistency
# dist[i][s] = 到状态 (节点 i, 第 s 个时间槽) 的最小费用
dist = [[INF] * (N_T + 1) for _ in range(101)] # 节点 1 到 100, 时间槽 0 到 N_T
prev = {}  # (i,s) -> (prev_i, prev_s, edge_id, bracket)

# 初始化起点
dist[start][0] = 0
hq = []  # Initialize the heap queue
heappush(hq, (0, start, 0)) # (cost, node_id, time_slot_idx)

# 构建邻接表 (主算法用)
adj = [[] for _ in range(101)] # 节点 1 到 100
for eid_adj, (u_adj, v_adj, vmax_adj) in enumerate(edges):
    adj[u_adj].append((eid_adj, v_adj, vmax_adj)) # Store edge_id, to_node, vmax (vmax might not be needed here if label_cost is primary)


print(f"Starting Dijkstra with T_max = {T_max:.4f} hours, N_T = {N_T} slots.")
processed_states = 0

# Dijkstra 主循环
while hq:
    c, i, s = heappop(hq)
    processed_states += 1

    if c > dist[i][s]: 
        continue
    
    # 目标检查可以放在这里，但通常对于有时间槽的Dijkstra，我们可能需要找到所有时间槽的最优解
    # 或者，如果只关心第一次到达，那可以提前break。
    # For this problem, we want the minimum cost path that reaches the goal *within* T_max.
    # The check `if (i==goal):` and `break` should be inside the loop,
    # but we need to ensure we are finding the *overall* minimum cost to goal within N_T slots.
    # The current backtracking logic correctly finds the minimum cost from all slots at the goal.

    # 遍历以 i 为起点的所有出边 (使用邻接表)
    # adj[i] contains tuples of (edge_id, to_node, vmax_of_edge)
    for eid_loop, v_edge, vmax_edge in adj[i]: 
        # eid_loop is the correct edge id from the original `edges` list
        for k_bracket, x_bracket_val in enumerate(BRACKETS):
            # Retrieve pre-calculated time and cost for this edge and speed bracket
            t_inc , cost_inc = label_cost.get((eid_loop, k_bracket), (INF, INF))

            if t_inc == INF: # Skip if this edge/bracket combination is impossible (e.g. div by zero speed)
                continue

            s2 = s + int(np.ceil(t_inc/SLOT)) # np.ceil 确保向上取整到下一个时间槽
            
            if s2 > N_T: 
                continue      # 超时，丢弃
            
            new_cost = c + cost_inc
            if new_cost < dist[v_edge][s2]:
                dist[v_edge][s2] = new_cost
                prev[(v_edge,s2)] = (i, s, eid_loop, k_bracket)
                heappush(hq, (new_cost, v_edge, s2))

print(f"Dijkstra finished. Processed {processed_states} states.")

# 5. 回溯得到 路线三 + 超速方案
s_goal_final = -1
min_total_cost_at_goal = INF # Renamed to avoid conflict with cost_on_segment etc.
for s_idx in range(N_T + 1): # Iterate up to N_T inclusive
    if dist[goal][s_idx] < min_total_cost_at_goal:
        min_total_cost_at_goal = dist[goal][s_idx]
        s_goal_final = s_idx

if s_goal_final == -1 or min_total_cost_at_goal == INF:
    print(f"Error: Goal {goal} was not reached within T_max = {T_max:.4f} hours ({N_T} slots).")
    # Check if goal was reached at all, even if exceeding T_max (already handled by dist initialization)
    min_cost_overall = INF
    s_overall = -1
    for s_chk_all in range(N_T + 1): # This loop is redundant if s_goal_final check is sufficient
        if dist[goal][s_chk_all] < min_cost_overall:
            min_cost_overall = dist[goal][s_chk_all]
            s_overall = s_chk_all
    if s_overall != -1:
         print(f"Minimum cost to reach goal {goal} is {min_cost_overall:.2f} at slot {s_overall} (which is {s_overall*SLOT:.2f}h), but this might not be the optimal if it exceeded T_max for the primary search.")
    else:
         print(f"Goal {goal} was not reached at all.")

else:
    print(f"Optimal solution: Cost = {min_total_cost_at_goal:.2f}, Time Slot = {s_goal_final} (Time = {s_goal_final*SLOT:.4f} hours)")
    # 5.2 回溯路径
    path_edges, speed_brackets = [], []
    curr_node, curr_slot = goal, s_goal_final
    
    temp_path_for_debug = []

    while (curr_node, curr_slot) != (start, 0):
        temp_path_for_debug.append((curr_node, curr_slot))
        if (curr_node, curr_slot) not in prev:
            print(f"Error in backtracking: State ({curr_node}, {curr_slot}) not found in prev map.")
            if curr_node == start and curr_slot == 0: # Should be caught by while condition
                 print("Current state is the start state (1,0), backtracking loop should have terminated.")
            path_edges = [] 
            speed_brackets = []
            break
        
        pre_i, pre_s, eid_path, k_path = prev[(curr_node, curr_slot)]
        path_edges.append(eid_path)
        speed_brackets.append(k_path)
        curr_node, curr_slot = pre_i, pre_s
    
    if not path_edges and not (start == goal and s_goal_final == 0) : 
        print("Path reconstruction resulted in an empty path.")
        if min_total_cost_at_goal != INF: # If a cost was found but path is empty
             print(f"Debug: Path to ({goal}, {s_goal_final}) with cost {min_total_cost_at_goal} could not be reconstructed.")
             print(f"Trace: {temp_path_for_debug}")
    else:
        # 5.3 还原节点序列（逆序→正序）
        route_nodes=[start]
        # Path_edges are in reverse order of travel. To get nodes in travel order:
        # for each edge_id in reversed(path_edges), the 'to_node' of that edge is the next node.
        for eid_val in reversed(path_edges):
            # edges[eid_val] is (from_node, to_node, v_max)
            # Ensure the from_node matches the previous node in route_nodes for sanity
            # This is implicitly handled by Dijkstra if prev map is correct.
            route_nodes.append(edges[eid_val][1]) # edges[eid][1] is the 'to_node'

        # 路线三
        print("\n路线三 (节点序列):")
        print(route_nodes)

        # 超速方案
        print("\n超速方案 (对应路段的超速倍率索引):")
        # speed_brackets is also in reverse order of travel, so [::-1] makes it correct travel order
        actual_speed_brackets = speed_brackets[::-1]
        print(actual_speed_brackets)

        print("\n超速方案详情:")
        total_time_precise = 0
        total_cost_recalculated = 0 # For verification
        
        # Iterate through the path in travel order
        # path_edges is currently reversed, so actual_path_edges = path_edges[::-1]
        actual_path_edges_ordered = path_edges[::-1]

        for i, eid_val in enumerate(actual_path_edges_ordered):
            from_node, to_node, _ = edges[eid_val] # Get edge details
            bracket_idx = actual_speed_brackets[i] # Get speed bracket for this segment
            speed_factor = BRACKETS[bracket_idx]
            time_taken, cost_on_segment = label_cost[(eid_val, bracket_idx)]
            
            total_time_precise += time_taken
            total_cost_recalculated += cost_on_segment
            print(f"  路段 {i+1} ({from_node} -> {to_node}), Edge ID {eid_val}: 超速倍率 x={speed_factor:.1f} (档位 {bracket_idx}), 耗时 {time_taken:.4f}h, 费用 {cost_on_segment:.2f}")
        
        print(f"\n总期望费用 (Dijkstra): {min_total_cost_at_goal:.2f}")
        print(f"总期望费用 (Recalculated from path): {total_cost_recalculated:.2f}")
        print(f"总精确计算时间: {total_time_precise:.4f} 小时")
        
        # 6. 验证与期望成本
        print("\n--- 验证 ---")
        print(f"T1 (给定基准时间): {T1:.4f} h") # Ensure T1 is formatted
        print(f"T_max (0.7 * T1): {T_max:.4f} h")
        print(f"计算得到的总时间: {total_time_precise:.4f} h (槽位时间: {s_goal_final*SLOT:.4f}h)")
        
        if total_time_precise <= T_max + 1e-9: # Add tolerance for float comparisons
            print("时间约束满足 (基于精确时间).")
        else:
            print(f"警告: 精确总时间 ({total_time_precise:.4f}h) 超过 T_max ({T_max:.4f}h)! 不符合约束。")
        
        if s_goal_final * SLOT <= T_max + 1e-9 :
            print("时间约束满足 (基于离散时间槽).")
        else:
            print(f"警告: 离散总时间 ({s_goal_final*SLOT:.4f}h) 超过 T_max ({T_max:.4f}h)! 不符合约束。")


        # Monte-Carlo simulation can be added here if needed

print("\n脚本执行完毕。")
