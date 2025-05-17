import pandas as pd
import numpy as np
import heapq

# 0. Constants
DISTANCE_PER_SEGMENT = 50.0  # km per edge
GAS_PRICE_PER_LITER = 7.76
ACCOMMODATION_COST_PER_HOUR = 20.0
ALPHA_OPTIONS = [0.0, 0.2, 0.5, 0.7] # Overspeeding percentages
PROB_CATCH = {0.0: 0.0, 0.2: 0.7, 0.5: 0.9, 0.7: 0.99} # Probability of being caught

# 1. 读取限速数据
# limits_col.csv: 纵向限速 (9x10)，limits_row.csv: 横向限速 (10x9)
# Ensure these files are in the same directory as the script, or provide full paths.
try:
    limits_col_df = pd.read_csv('D:\Code\mm\data\limits_col.csv', header=None)
    limits_row_df = pd.read_csv('D:\Code\mm\data\limits_row.csv', header=None)
except FileNotFoundError:
    print("Error: Ensure 'limits_col.csv' and 'limits_row.csv' are in the current directory.")
    exit()

# 2. 路线一 (Route One) 节点定义
# Route One 节点索引 (1-indexed): [1,2,12,13,...,100]
ROUTE_ONE_NODES_1_INDEXED = [1, 2, 12, 13, 14, 24, 25, 35, 45, 55, 56, 57, 58, 59, 69, 79, 89, 90, 100]

# Helper: Node ID (1-100) to 0-indexed grid coordinates (row, col)
def node_to_coords(node_id_1_indexed):
    # grid_row 0 (bottom) to 9 (top)
    # grid_col 0 (left) to 9 (right)
    return (node_id_1_indexed - 1) // 10, (node_id_1_indexed - 1) % 10

# Helper: 0-indexed grid coordinates to Node ID (1-100)
def coords_to_node(r, c):
    return r * 10 + c + 1

# Helper: Get speed limit for a segment
def get_speed_limit(u_node_1_idx, v_node_1_idx, limits_row, limits_col):
    r_u, c_u = node_to_coords(u_node_1_idx)
    r_v, c_v = node_to_coords(v_node_1_idx)

    if r_u == r_v:  # Horizontal movement
        # limits_row.iloc[grid_row, grid_col_start_node]
        return limits_row.iloc[r_u, min(c_u, c_v)]
    elif c_u == c_v:  # Vertical movement
        # limits_col.iloc[grid_row_start_node, grid_col]
        return limits_col.iloc[min(r_u, r_v), c_u]
    else:
        raise ValueError(f"Nodes {u_node_1_idx} and {v_node_1_idx} are not adjacent.")

# 3. 罚款函数
def calculate_fine_amount(speed_limit, actual_speed):
    if actual_speed <= speed_limit:
        return 0
    
    over_pct = (actual_speed - speed_limit) / speed_limit

    if speed_limit < 50:
        if over_pct <= 0.2: return 50
        if over_pct <= 0.5: return 100
        if over_pct <= 0.7: return 300
        return 500
    elif speed_limit <= 80:
        if over_pct <= 0.2: return 100
        if over_pct <= 0.5: return 150
        if over_pct <= 0.7: return 500
        return 1000
    elif speed_limit <= 100:
        if over_pct <= 0.2: return 150
        if over_pct <= 0.5: return 200
        if over_pct <= 0.7: return 1000
        return 1500
    else: # speed_limit > 100 (e.g., 120)
        # Problem description for >100km/h:
        # ≤ 50% → 200
        # ≤ 70% → 1500
        # > 70% → 2000
        # Note: The problem statement's fine structure for >100km/h has a gap if over_pct is exactly 0.
        # Assuming "≤ 50%" means (0, 0.5].
        if over_pct <= 0.5: return 200 # This covers up to 50% overspeed for 120km/h limit
        if over_pct <= 0.7: return 1500
        return 2000


# Helper: Calculate segment costs (time, fuel, toll, accommodation, expected fine)
def calculate_segment_costs(speed_limit, actual_speed, distance, alpha_overspeed):
    time = distance / actual_speed if actual_speed > 0 else float('inf')
    
    # Fuel consumption V = 0.0625v + 1.875 (L/100km)
    fuel_consumption_rate = 0.0625 * actual_speed + 1.875
    fuel_cost = fuel_consumption_rate * (distance / 100.0) * GAS_PRICE_PER_LITER
    
    # Toll cost (0.5 yuan/km for highways with limit 120)
    toll_cost = 0.5 * distance if speed_limit == 120 else 0.0
    
    accommodation_cost = ACCOMMODATION_COST_PER_HOUR * time
    
    fine_amount = calculate_fine_amount(speed_limit, actual_speed)
    expected_fine = fine_amount * PROB_CATCH.get(alpha_overspeed, 0.0)
    
    total_cost = fuel_cost + toll_cost + accommodation_cost + expected_fine
    return time, total_cost, fuel_cost, toll_cost, accommodation_cost, expected_fine

# 4. 计算路线一在不超速情况下的基础时间和总费用
#    (Calculate base time and total cost for Route One without speeding)
def compute_route_one_base_info(route_1_nodes, dist_segment, limits_r, limits_c):
    T0_R1 = 0.0
    C0_total_R1_no_speeding = 0.0
    
    route_one_segments_details = []

    for i in range(len(route_1_nodes) - 1):
        u_node = route_1_nodes[i]
        v_node = route_1_nodes[i+1]
        
        L_segment = get_speed_limit(u_node, v_node, limits_r, limits_c)
        actual_speed_segment = L_segment # No speeding
        
        time_s, total_cost_s, fuel_s, toll_s, accom_s, fine_s = calculate_segment_costs(
            L_segment, actual_speed_segment, dist_segment, 0.0
        )
        
        T0_R1 += time_s
        C0_total_R1_no_speeding += total_cost_s
        
        route_one_segments_details.append({
            'u': u_node, 'v': v_node, 'L': L_segment,
            't0': time_s, 'cost0_total': total_cost_s,
            'fuel0': fuel_s, 'toll0': toll_s # Toll is fixed for a segment regardless of speed
        })
        
    return T0_R1, C0_total_R1_no_speeding, route_one_segments_details


# 5. 计算其他所有路径的最小总费用（不超速）- Dijkstra
#    (Calculate min total cost for any other path, no speeding - Dijkstra)
def calculate_min_cost_other_path_dijkstra(limits_r, limits_c, dist_segment, start_node_id=1, end_node_id=100):
    num_nodes = 100
    adj = [[] for _ in range(num_nodes + 1)] # 1-indexed

    for r_idx in range(10): # 0-9 for grid rows
        for c_idx in range(10): # 0-9 for grid cols
            current_node = coords_to_node(r_idx, c_idx)
            
            # Potential neighbors: (dr, dc)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r_idx + dr, c_idx + dc
                
                if 0 <= nr <= 9 and 0 <= nc <= 9:
                    neighbor_node = coords_to_node(nr, nc)
                    speed_lim = get_speed_limit(current_node, neighbor_node, limits_r, limits_c)
                    actual_spd = speed_lim # No speeding
                    
                    _t, cost_seg, _f, _to, _a, _fi = calculate_segment_costs(
                        speed_lim, actual_spd, dist_segment, 0.0
                    )
                    adj[current_node].append((neighbor_node, cost_seg))

    min_costs = {i: float('inf') for i in range(1, num_nodes + 1)}
    min_costs[start_node_id] = 0
    pq = [(0, start_node_id)] # (cost, node)

    while pq:
        cost, u = heapq.heappop(pq)

        if cost > min_costs[u]:
            continue
        if u == end_node_id: # Optimization: stop if end_node is reached
            break 

        for v, weight in adj[u]:
            if min_costs[u] + weight < min_costs[v]:
                min_costs[v] = min_costs[u] + weight
                heapq.heappush(pq, (min_costs[v], v))
    
    return min_costs[end_node_id]


# --- Main Computations ---
T0_R1, C0_total_R1_no_speeding, route_one_segment_params = compute_route_one_base_info(
    ROUTE_ONE_NODES_1_INDEXED, DISTANCE_PER_SEGMENT, limits_row_df, limits_col_df
)

print(f"Route One - Base Time (T0_R1): {T0_R1:.4f} hours")
print(f"Route One - Base Total Cost (C0_total_R1_no_speeding): {C0_total_R1_no_speeding:.4f}元")

C_min_other_path_no_speeding = calculate_min_cost_other_path_dijkstra(
    limits_row_df, limits_col_df, DISTANCE_PER_SEGMENT
)
print(f"Min cost of any other path (no speeding): {C_min_other_path_no_speeding:.4f}元")


# 6. 计算背包容量 B (Knapsack Capacity B)
# B is the max additional cost Route One can incur while remaining optimal
B_knapsack_capacity = C_min_other_path_no_speeding - C0_total_R1_no_speeding
print(f"Knapsack capacity B: {B_knapsack_capacity:.4f}元")

if B_knapsack_capacity < 0:
    print("Warning: Route One is already more expensive than an alternative path without any speeding.")
    # Depending on problem interpretation, this might mean no speeding is allowed,
    # or the problem assumes B will be non-negative.
    # For now, we proceed, but DP might yield no savings if B is too small or negative.

# 7. 计算每条边在各超速级别下的节省时间 (Delta_t) 和 费用增量 (Delta_C_knapsack)
#    (Calculate Delta_t and Delta_C_knapsack for each segment and overspeed option)
num_segments_R1 = len(route_one_segment_params)
Delta_t_R1 = np.zeros((num_segments_R1, len(ALPHA_OPTIONS)))
# Delta_C_knapsack is the "weight" for the knapsack: (Fuel_k-Fuel_0) + Fine_k - 20*Delta_t
Delta_C_knapsack_R1 = np.zeros((num_segments_R1, len(ALPHA_OPTIONS)))

for i, seg_param in enumerate(route_one_segment_params):
    L_seg = seg_param['L']
    t0_seg = seg_param['t0']
    fuel0_seg = seg_param['fuel0']
    # toll0_seg is fixed for the segment, so (Toll_k - Toll_0) = 0

    for j, alpha_k in enumerate(ALPHA_OPTIONS):
        actual_speed_k = L_seg * (1 + alpha_k)
        
        tk_seg, _, fuel_k_seg, _, _, fine_k_seg_expected = calculate_segment_costs(
            L_seg, actual_speed_k, DISTANCE_PER_SEGMENT, alpha_k
        )
        
        time_saved_seg = t0_seg - tk_seg
        Delta_t_R1[i, j] = time_saved_seg
        
        # Change in (Fuel + Fine)
        delta_fuel_fine_seg = (fuel_k_seg - fuel0_seg) + fine_k_seg_expected
        
        # Knapsack weight: change in (Fuel + Fine) - change in (Accommodation cost due to time saved)
        Delta_C_knapsack_R1[i, j] = delta_fuel_fine_seg - ACCOMMODATION_COST_PER_HOUR * time_saved_seg

# 8. 多选分组背包 DP (Multiple-choice Knapsack DP)
# dp[i][c] = Max time saved using first i segments with total knapsack weight c
# Scale costs for DP table indexing
COST_SCALE_FACTOR = 100 
# Ensure B_knapsack_capacity_int is non-negative for dp table size
B_knapsack_capacity_int = int(np.floor(max(0, B_knapsack_capacity) * COST_SCALE_FACTOR))

# Initialize DP table with a value indicating not reachable or no savings
# Using -1 as float('inf') can cause issues with additions if not handled carefully
dp_table = np.full((num_segments_R1 + 1, B_knapsack_capacity_int + 1), -1.0) 
choice_table = np.zeros((num_segments_R1 + 1, B_knapsack_capacity_int + 1), dtype=int)
dp_table[0, 0] = 0.0 # Base case: 0 segments, 0 cost, 0 time saved

for i in range(1, num_segments_R1 + 1): # For each segment
    for c_scaled in range(B_knapsack_capacity_int + 1): # For each possible scaled knapsack cost
        for k_opt_idx, alpha_k_val in enumerate(ALPHA_OPTIONS): # For each overspeeding option
            
            # Knapsack weight for current segment (i-1) and option (k_opt_idx)
            w_ik_scaled = int(round(Delta_C_knapsack_R1[i-1, k_opt_idx] * COST_SCALE_FACTOR))
            time_saved_ik = Delta_t_R1[i-1, k_opt_idx]
            
            prev_c_scaled = c_scaled - w_ik_scaled
            
            if prev_c_scaled >= 0 and dp_table[i-1, prev_c_scaled] > -0.5: # Check if previous state is valid (>-1)
                current_total_time_saved = dp_table[i-1, prev_c_scaled] + time_saved_ik
                if current_total_time_saved > dp_table[i, c_scaled]:
                    dp_table[i, c_scaled] = current_total_time_saved
                    choice_table[i, c_scaled] = k_opt_idx

# 9. 回溯最优方案 (Backtrack for the optimal solution)
max_total_time_saved = -1.0
best_cost_scaled_idx = -1

# Find the max time saved within the knapsack capacity
for c_idx in range(B_knapsack_capacity_int + 1):
    if dp_table[num_segments_R1, c_idx] > max_total_time_saved:
        max_total_time_saved = dp_table[num_segments_R1, c_idx]
        best_cost_scaled_idx = c_idx

chosen_alphas_pct = []
if best_cost_scaled_idx != -1 and max_total_time_saved > -0.5 : # A valid solution was found
    current_c_scaled = best_cost_scaled_idx
    for i_seg in range(num_segments_R1, 0, -1):
        option_idx = choice_table[i_seg, current_c_scaled]
        chosen_alphas_pct.append(ALPHA_OPTIONS[option_idx] * 100)
        
        w_chosen_scaled = int(round(Delta_C_knapsack_R1[i_seg-1, option_idx] * COST_SCALE_FACTOR))
        current_c_scaled -= w_chosen_scaled
    chosen_alphas_pct.reverse()
else: # No way to save time under the budget B (or B was negative/zero)
    max_total_time_saved = 0.0 # No time saved
    chosen_alphas_pct = [0.0] * num_segments_R1 # Default to no overspeeding
    best_cost_scaled_idx = 0 # Corresponds to sum of knapsack weights being 0

# 10. 输出结果 (Output results)
actual_sum_knapsack_weights = best_cost_scaled_idx / COST_SCALE_FACTOR
final_total_cost_R1 = C0_total_R1_no_speeding + actual_sum_knapsack_weights

print(f"\n--- Results ---")
print(f"最大可降低时间 (Max time reduction): {max_total_time_saved:.4f} 小时 (hours)")
print(f"此时总费用 (Total cost at this time reduction): {final_total_cost_R1:.4f} 元 (yuan)")

print("\n各路段超速方案 (Overspeeding plan per segment for Route One):")
for idx, alpha_pct_val in enumerate(chosen_alphas_pct):
    seg_info = route_one_segment_params[idx]
    print(f"路段 {seg_info['u']}→{seg_info['v']} (限速 {seg_info['L']} km/h): 超速 {alpha_pct_val:.0f}%")

