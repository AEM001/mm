import numpy as np
from heapq import heappush, heappop

# ---------- 常量 ----------
DIST        = 50.0    # 每段距离（km）
FUEL_PRICE  = 7.76    # 汽油单价（元/升）
TOLL_PER_KM = 0.5     # 高速费（元/公里）
TIME_RATE   = 20.0    # 餐饮住宿游览费率（元/小时）

# ---------- 罚款与探测概率 ----------
def detect_prob(pct):
    if pct <= 0:   return 0.0
    elif pct <= 20: return 0.70
    elif pct <= 50: return 0.90
    else:           return 0.99  # pct ≤ 70

def fine(speed_limit, pct):
    if pct <= 0: return 0
    if speed_limit < 50:
        return 50   if pct <= 20 else \
               100  if pct <= 50 else \
               300  if pct <= 70 else 500
    if speed_limit <= 80:
        return 100  if pct <= 20 else \
               150  if pct <= 50 else \
               500  if pct <= 70 else 1000
    if speed_limit <= 100:
        return 150  if pct <= 20 else \
               200  if pct <= 50 else \
               1000 if pct <= 70 else 1500
    # speed_limit > 100
    return 200   if pct <= 50 else \
           1500  if pct <= 70 else 2000

# ---------- 限速矩阵 (行列互换 for 列向 limits_col) ----------
limits_col = np.array([
 [60,90,60,60,40,40,40,60,60,120],
 [40,60,60,60,40,90,60,60,60,120],
 [40,60,60,60,120,90,60,40,40,120],
 [40,60,90,60,120,60,60,60,40,120],
 [60,60,90,60,120,60,60,60,40,40],
 [60,40,90,60,120,60,40,40,40,40],
 [40,40,90,40,120,40,40,40,90,40],
 [60,60,90,40,40,60,60,60,90,60],
 [60,60,90,60,60,60,90,60,60,90],
])  # shape (9,10)

limits_row = np.array([
 [90,40,40,40,40,40,40,40,40],
 [60,90,90,40,40,60,60,40,40],
 [40,60,60,60,60,60,40,40,40],
 [40,60,90,90,90,60,60,60,60],
 [60,60,60,60,60,60,60,60,40],
 [60,60,90,90,90,90,90,90,60],
 [40,40,60,60,40,40,40,60,40],
 [60,60,40,40,40,40,90,90,90],
 [120,120,120,120,40,40,90,90,90],
 [60,60,60,60,40,60,60,60,60],
])  # shape (10,9)

# ---------- 工具：节点编号与单段成本 ----------
def node(r, c):
    """0-based 编号：r=0..9, c=0..9 → id=0..99"""
    return r*10 + c

def cost_time(actual_v, speed_limit):
    """
    返回：(总成本, 时间h)，
    成本包含：时间成本 + 燃油费 + 通行费 + 期望罚款
    """
    t = DIST / actual_v
    time_fee = TIME_RATE * t
    fuel_L = (0.0625*actual_v + 1.875) * (DIST/100.0)
    fuel_fee = fuel_L * FUEL_PRICE
    toll = TOLL_PER_KM * DIST if speed_limit == 120 else 0.0
    pct = (actual_v - speed_limit) / speed_limit * 100
    exp_penalty = detect_prob(pct) * fine(speed_limit, pct)
    return time_fee + fuel_fee + toll + exp_penalty, t

# ---------- Dijkstra (返回最小费用) ----------
def dijkstra(adj, src, dst):
    INF = 1e30
    dist = [INF]*100
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d,u = heappop(pq)
        if d > dist[u]: continue
        if u == dst: break
        for v,w in adj[u]:
            nd = d + w
            if nd + 1e-12 < dist[v]:
                dist[v] = nd
                heappush(pq, (nd, v))
    return dist[dst]

# ---------- 构建“不超速”图 H，计算 baseline_min_other ----------
H = [[] for _ in range(100)]
# 纵向边（limits_col 行列互换）
for r in range(9):
    for c in range(10):
        u, v = node(r, c), node(r+1, c)
        s = limits_col[r, c]
        w, _ = cost_time(s, s)
        H[u].append((v, w)); H[v].append((u, w))
# 横向边
for r in range(10):
    for c in range(9):
        u, v = node(r, c), node(r, c+1)
        s = limits_row[r, c]
        w, _ = cost_time(s, s)
        H[u].append((v, w)); H[v].append((u, w))

baseline_min_other = round(dijkstra(H, 0, 99), 4)

# ---------- 定义“路线一”及其基准成本/时间 ----------
route1 = [1,2,12,13,14,24,25,35,45,55,56,57,58,59,69,79,89,90,100]
# 转成 0-based
route1 = [n-1 for n in route1]
segments = list(zip(route1[:-1], route1[1:]))

# 预生成边限速查表
edge_limit = {}
# 纵向
for r in range(9):
    for c in range(10):
        u, v = node(r,c), node(r+1,c)
        s = limits_col[r, c]
        edge_limit[(u,v)] = edge_limit[(v,u)] = s
# 横向
for r in range(10):
    for c in range(9):
        u, v = node(r,c), node(r,c+1)
        s = limits_row[r, c]
        edge_limit[(u,v)] = edge_limit[(v,u)] = s

# 计算 base_cost / base_time
base_cost = base_time = 0.0
for u, v in segments:
    s = edge_limit[(u,v)]
    c0, t0 = cost_time(s, s)
    base_cost += c0
    base_time += t0

base_cost  = round(base_cost, 4)
base_time  = round(base_time, 4)

# ---------- 预算与超速选项 ----------
# 允许的最大超速比例集合
deltas = [0, 20, 50, 70]

# 生成每段的 (delta_pct, cost_diff, time_saved, real_speed)
choices = []
for u, v in segments:
    s_lim = edge_limit[(u,v)]
    c0, t0 = cost_time(s_lim, s_lim)
    opts = []
    for dg in deltas:
        v_real = s_lim * (1 + dg/100.0)
        c1, t1 = cost_time(v_real, s_lim)
        dc = round(c1 - c0, 8)
        dt = round(t0 - t1, 8)
        opts.append((dg, dc, dt, v_real))
    choices.append(opts)

# 预算：使得总成本 ≤ baseline_min_other
# 即 base_cost + sum(dc_i) ≤ baseline_min_other
budget = baseline_min_other - base_cost  # 可以为正，也可以为负

# ---------- DP: 最优时间节省 ----------
# states: dict{ sum_dc: (sum_dt, plan_list) }
# plan_list 为依次选的 (delta_pct, dc, dt, real_speed)
states = { 0.0: (0.0, []) }

for opts in choices:
    nxt = {}
    for sum_dc, (sum_dt, plan) in states.items():
        for dg, dc, dt, vr in opts:
            new_dc = sum_dc + dc
            # 约束：到目前为止 cost_diff = new_dc，最终需 new_dc ≤ budget
            # 如果 budget>=0，则可提前剪枝；否则不剪枝，等所有段后再判断
            if budget >= 0 and new_dc > budget + 1e-9:
                continue
            new_dt = sum_dt + dt
            new_plan = plan + [(dg, dc, dt, vr)]
            key = round(new_dc, 4)
            # 取最大的 time_saved
            prev = nxt.get(key)
            if prev is None or new_dt > prev[0] + 1e-12:
                nxt[key] = (new_dt, new_plan)
    # 剪除被支配状态
    pruned = {}
    best_dt = -1.0
    for k in sorted(nxt):
        dt_val, pl = nxt[k]
        if dt_val > best_dt + 1e-12:
            pruned[k] = (dt_val, pl)
            best_dt = dt_val
    states = pruned
    if not states:
        break

# ---------- 选出最优方案 ----------
best_dt = -1e9
best_dc = None
best_plan = None

for sum_dc, (sum_dt, plan) in states.items():
    # 对 budget<0 的情况，需在最后判断 sum_dc ≤ budget
    if sum_dc <= budget + 1e-9:
        if sum_dt > best_dt:
            best_dt = sum_dt
            best_dc = sum_dc
            best_plan = plan

if best_plan is None:
    # 无可行方案，则保持限速
    best_dc = 0.0
    best_dt = 0.0
    best_plan = [(0, 0.0, 0.0, edge_limit[(u,v)]) for (u,v) in segments]

# 计算最终成本与时间
final_cost = round(base_cost + best_dc, 4)
final_time = round(base_time - best_dt, 4)

# ---------- 输出 ----------
print(f"基准（限速）费用      : {base_cost:.4f} 元")
print(f"基准（限速）时间      : {base_time:.4f} 小时")
print(f"其他路径最小费用      : {baseline_min_other:.4f} 元")
print(f"\n===== 结果 =====")
print(f"最短可行时间         : {final_time:.4f} 小时")
print(f"对应总费用           : {final_cost:.4f} 元\n")
print("路线一超速方案 (u->v | 限速 | 超速% | 实际速 | Δcost   | Δtime)")
for (u,v), (dg, dc, dt, vr) in zip(segments, best_plan):
    s_lim = edge_limit[(u,v)]
    print(f"{u+1:3d}->{v+1:<3d} | {s_lim:3d}km/h | {dg:3d}% | {vr:7.2f}km/h | {dc:+8.4f} | {dt:+8.4f}")
