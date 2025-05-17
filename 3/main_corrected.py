import pandas as pd
import numpy as np
import heapq # 新增导入
import os    # 新增导入

# 1. 读取限速数据
# 使用 os.path 确保路径的正确性
script_dir = os.path.dirname(os.path.abspath(__file__))
limits_col_path = os.path.join(script_dir, '..', 'data', 'limits_col.csv')
limits_row_path = os.path.join(script_dir, '..', 'data', 'limits_row.csv')

limits_col_df = pd.read_csv(limits_col_path, header=None)
limits_row_df = pd.read_csv(limits_row_path, header=None)

# 2. 构建图G，仅关注路线一上的节点和边属性
# Route One 节点索引 (1-indexed): [1,2,12,13,...,100]
route_nodes = [1,2,12,13,14,24,25,35,45,55,56,57,58,59,69,79,89,90,100]
# 计算边的限速 L_list，与是否收费
edges_route_one = []  # 列表元素: dict{name,u,v,limit,toll_per_km}
for i in range(len(route_nodes)-1):
    u = route_nodes[i]
    v = route_nodes[i+1]
    if v == u + 1: # 横向
        r_0idx = (u-1)//10
        c_0idx = (u-1)%10
        L = limits_row_df.iloc[r_0idx, c_0idx]
    else: # 纵向 v == u + 10
        r_0idx = (u-1)//10
        c_0idx = (u-1)%10
        L = limits_col_df.iloc[r_0idx, c_0idx] # Corrected indexing based on how nodes map to CSV
    toll = 0.5 if L == 120 else 0.0
    edges_route_one.append({'u':u, 'v':v, 'L':L, 'toll':toll})

d_segment = 50.0  # km per edge
alpha_options = [0.0, 0.2, 0.5, 0.7]
prob_catch_single_radar = {0.0:0.0, 0.2:0.7, 0.5:0.9, 0.7:0.99} # Probability of being caught by ONE radar

TOTAL_SEGMENTS_IN_GRID = 180 # 10*9 horizontal + 9*10 vertical (actually 10*9 vert for 9x10 file)
# Total segments: 10 rows * 9 horiz/row + 10 cols * 9 vert/col = 90 + 90 = 180 unique directed segments if we consider one way.
# Or, number of physical road pieces = 10*9 + 9*10 = 180.
RHO_MOBILE_RADAR = 20.0 / TOTAL_SEGMENTS_IN_GRID


# 3. 罚款函数
def calc_fine(L, alpha):
    over_pct = alpha * 100
    if over_pct <= 0:
        return 0
    if L < 50:
        if over_pct <= 20: return 50
        if over_pct <= 50: return 100
        if over_pct <= 70: return 300
        return 500
    elif L <= 80:
        if over_pct <= 20: return 100
        if over_pct <= 50: return 150
        if over_pct <= 70: return 500
        return 1000
    elif L <= 100:
        if over_pct <= 20: return 150
        if over_pct <= 50: return 200
        if over_pct <= 70: return 1000
        return 1500
    else:
        if over_pct <= 50: return 200
        if over_pct <= 70: return 1500
        return 2000

# 4. 计算不超速时的总时间T0和总费用C0
def compute_base_cost_time(route_edges_list, segment_distance):
    T_base = 0.0
    C_base = 0.0
    # 餐饮住宿游览费 c=20t（元）
    # 汽车速度为v (公里/小时)时，每百公里耗油量V=0.0625v+1.875（升）
    # 汽油单价均为7.76元/升
    # 高速公路 L=120, 每公里收费0.5元
    for e in route_edges_list:
        L = float(e['L'])
        if L == 0: continue # Should not happen for a valid path
        
        v = L
        t = segment_distance / v
        T_base += t
        
        # 餐饮住宿游览费
        C_base += 20 * t
        
        # 燃油费
        V_fuel = 0.0625 * v + 1.875
        C_base += V_fuel * (segment_distance / 100.0) * 7.76
        
        # 通行费
        C_base += e['toll'] * segment_distance # e['toll'] is per_km
    return T_base, C_base

# 5. 计算其他所有路径的最小总费用（不超速），得到 C_min
def dijkstra_full_graph(limits_row, limits_col, num_nodes=100, start_node_1idx=1, end_node_1idx=100, segment_d=50.0):
    adj = {i: [] for i in range(1, num_nodes + 1)}

    def get_segment_cost_no_speeding(L_val, toll_per_km_val, dist_val):
        if L_val == 0: return float('inf')
        v_val = float(L_val)
        t_val = dist_val / v_val
        
        cost_val = 20 * t_val # Time-based cost
        
        V_fuel_val = 0.0625 * v_val + 1.875
        cost_val += V_fuel_val * (dist_val / 100.0) * 7.76 # Fuel cost
        
        cost_val += toll_per_km_val * dist_val # Toll cost
        return cost_val

    for r_idx in range(10):  # 0-9 grid row
        for c_idx in range(10): # 0-9 grid col
            u_node = r_idx * 10 + c_idx + 1

            # Horizontal edge: u_node to u_node+1
            if c_idx < 9:
                v_node_h = u_node + 1
                L_h = limits_row.iloc[r_idx, c_idx]
                toll_h = 0.5 if L_h == 120 else 0.0
                cost_h = get_segment_cost_no_speeding(L_h, toll_h, segment_d)
                if cost_h != float('inf'):
                    adj[u_node].append((v_node_h, cost_h))
                    adj[v_node_h].append((u_node, cost_h)) # Bidirectional

            # Vertical edge: u_node to u_node+10
            if r_idx < 9: # Max r_idx for vertical segment start is 8 (for 9 rows in limits_col)
                v_node_v = u_node + 10
                L_v = limits_col.iloc[r_idx, c_idx] # limits_col is 9x10, row index r_idx (0-8), col index c_idx (0-9)
                toll_v = 0.5 if L_v == 120 else 0.0
                cost_v = get_segment_cost_no_speeding(L_v, toll_v, segment_d)
                if cost_v != float('inf'):
                    adj[u_node].append((v_node_v, cost_v))
                    adj[v_node_v].append((u_node, cost_v)) # Bidirectional
    
    min_costs_to_node = {i: float('inf') for i in range(1, num_nodes + 1)}
    min_costs_to_node[start_node_1idx] = 0
    pq = [(0, start_node_1idx)] # (cost, node)

    while pq:
        current_min_cost, u = heapq.heappop(pq)

        if current_min_cost > min_costs_to_node[u]:
            continue
        if u == end_node_1idx: # Optimization: if we only need cost to end_node
            return min_costs_to_node[end_node_1idx]

        for v_neighbor, weight in adj[u]:
            if min_costs_to_node[u] + weight < min_costs_to_node[v_neighbor]:
                min_costs_to_node[v_neighbor] = min_costs_to_node[u] + weight
                heapq.heappush(pq, (min_costs_to_node[v_neighbor], v_neighbor))
                
    return min_costs_to_node[end_node_1idx]

C_min_alternative_paths = dijkstra_full_graph(limits_row_df, limits_col_df, segment_d=d_segment)

# 6. 计算背包容量 B
T0, C0 = compute_base_cost_time(edges_route_one, d_segment)
B = C_min_alternative_paths - C0 # This B can be negative

# 7. 计算每条边在各超速级别下的增量 Δt, ΔC
n_segments_route_one = len(edges_route_one)
Delta_t = np.zeros((n_segments_route_one, len(alpha_options)))
Delta_C = np.zeros((n_segments_route_one, len(alpha_options)))

for i, e in enumerate(edges_route_one):
    L = float(e['L'])
    toll_per_km = e['toll']
    
    # 基准 (不超速)
    v0 = L
    t0 = d_segment / v0
    
    cost_time0 = 20 * t0
    V_fuel0 = 0.0625 * v0 + 1.875
    cost_fuel0 = V_fuel0 * (d_segment / 100.0) * 7.76
    cost_toll0 = toll_per_km * d_segment
    cost0 = cost_time0 + cost_fuel0 + cost_toll0
    
    for j, alpha in enumerate(alpha_options):
        v = L * (1+alpha)
        tk = d_segment / v
        time_costk = 20 * tk
        Vk = 0.0625 * v + 1.875
        fuel_costk = Vk * (d_segment / 100.0) * 7.76
        toll_costk = toll_per_km * d_segment
        costk = time_costk + fuel_costk + toll_costk
        
        # 罚款期望
        fine = calc_fine(L, alpha) * prob_catch_single_radar.get(alpha, 0)
        
        # 增量
        Delta_t[i,j] = t0 - tk
        Delta_C[i,j] = (costk + fine) - cost0

# 8. 多选分组背包 DP
# dp[i][c] = 在前i条边消耗增量费用不超过c时，最大节省时间
# 这里将费用单位扩大100倍并转化为整数索引
scale = 100
B_int = int(np.floor(B*scale))
dp = np.full((n_segments_route_one+1, B_int+1), -np.inf)
choice = np.zeros((n_segments_route_one+1, B_int+1), dtype=int)
dp[0,0] = 0
for i in range(1, n_segments_route_one+1):
    for c in range(B_int+1):
        # 默认不超速
        dp[i,c] = dp[i-1,c]
        choice[i,c] = 0
        
        for j, alpha in enumerate(alpha_options):
            cost_inc = int(np.round(Delta_C[i-1,j]*scale))
            if c >= cost_inc and dp[i-1,c-cost_inc] > -np.inf:
                val = dp[i-1,c-cost_inc] + Delta_t[i-1,j]
                if val > dp[i,c]:
                    dp[i,c] = val
                    choice[i,c] = j

# 9. 回溯最优方案
c_best = np.argmax(dp[n_segments_route_one])
t_best = dp[n_segments_route_one,c_best]
alpha_sel = []
c = c_best
for i in range(n_segments_route_one, 0, -1):
    j = choice[i,c]
    alpha_sel.append(alpha_options[j]*100)
    c -= int(np.round(Delta_C[i-1,j]*scale))
alpha_sel.reverse()

# 10. 输出结果
print(f"最大可降低时间: {t_best:.4f} 小时")
print(f"此时总费用: {C0 + c_best/scale:.4f} 元")
for idx, pct in enumerate(alpha_sel, start=1):
    e = edges_route_one[idx-1]
    print(f"路段 {e['u']}→{e['v']}: 超速 {pct:.0f}%")
