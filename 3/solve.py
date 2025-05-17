"""
题目三：在允许 0/20/50/70% 超速的情况下，
        求“路线一”保持费用最优时可压缩的最短时间
"""

from heapq import heappush, heappop

# ---------- 常量 ----------
DIST = 50.0                  # km
FUEL_PRICE = 7.76            # 元/L
TOLL_PER_KM = 0.5            # 高速费
TIME_RATE = 20.0             # 餐饮住宿游览费 20 元/小时

# ---------- 罚款和被探测概率 ----------
def detect_prob(pct):
    if pct <= 0:   return 0.0
    if pct <= 20:  return 0.70
    if pct <= 50:  return 0.90
    return 0.99            # 50–70

def fine(speed_limit, pct):
    if pct <= 0: return 0
    if speed_limit < 50:
        return 50 if pct <= 20 else 100 if pct <= 50 else 300 if pct <= 70 else 500
    if speed_limit <= 80:
        return 100 if pct <= 20 else 150 if pct <= 50 else 500 if pct <= 70 else 1000
    if speed_limit <= 100:
        return 150 if pct <= 20 else 200 if pct <= 50 else 1000 if pct <= 70 else 1500
    # >100
    return 200 if pct <= 50 else 1500 if pct <= 70 else 2000

# ---------- 限速矩阵 ----------
limits_col = [  # 9×10  纵向
 [60,90,60,60,40,40,40,60,60,120],
 [40,60,60,60,40,90,60,60,60,120],
 [40,60,60,60,120,90,60,40,40,120],
 [40,60,90,60,120,60,60,60,40,120],
 [60,60,90,60,120,60,60,60,40,40],
 [60,40,90,60,120,60,40,40,40,40],
 [40,40,90,40,120,40,40,40,90,40],
 [60,60,90,40,40,60,60,60,90,60],
 [60,60,90,60,60,60,90,60,60,90],
]
limits_row = [  # 10×9  横向
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
]

# ---------- 工具函数 ----------
def node(r, c): return r*10 + c

def cost_time(actual_v, s_lim):
    """返回 (费用,时间h)"""
    t = DIST / actual_v
    time_fee = TIME_RATE * t
    fuel_L = (0.0625*actual_v + 1.875) * (DIST/100)
    fuel_fee = fuel_L * FUEL_PRICE
    toll = TOLL_PER_KM*DIST if s_lim==120 else 0
    pct = (actual_v - s_lim)/s_lim*100
    exp_penalty = detect_prob(pct)*fine(s_lim, pct)
    return time_fee + fuel_fee + toll + exp_penalty, t

# ---------- 构建“扩展”图 (4 倍边) ----------
G = [[] for _ in range(100)]
for r in range(9):          # vertical
    for c in range(10):
        u, v = node(r,c), node(r+1,c)
        s = limits_col[r][c]
        for pct in (0,20,50,70):
            vv = s*(1+pct/100)
            cst, _ = cost_time(vv, s)
            G[u].append((v, cst))
            G[v].append((u, cst))
for r in range(10):         # horizontal
    for c in range(9):
        u, v = node(r,c), node(r,c+1)
        s = limits_row[r][c]
        for pct in (0,20,50,70):
            vv = s*(1+pct/100)
            cst, _ = cost_time(vv, s)
            G[u].append((v, cst))
            G[v].append((u, cst))

def dijkstra(src, dst):
    INF = 1e30
    dist = [INF]*100
    dist[src] = 0
    pq=[(0,src)]
    while pq:
        d,u = heappop(pq)
        if d>dist[u]: continue
        if u==dst: break
        for v,w in G[u]:
            nd = d+w
            if nd < dist[v]-1e-9:
                dist[v]=nd
                heappush(pq,(nd,v))
    return dist[dst]

C_min = round(dijkstra(0,99), 4)        # 四舍五入到 4 位后作为基准

# ---------- 路线一基准 (不超速) ----------
route1 = [0,1,11,12,13,23,24,34,44,54,55,56,57,58,68,78,88,89,99]
segments = list(zip(route1[:-1], route1[1:]))

base_cost, base_time = 0.0, 0.0
edge_speedlim = {}
# 生成 speed_limit 字典以便后用
for r in range(9):
    for c in range(10):
        edge_speedlim[(node(r,c), node(r+1,c))] = edge_speedlim[(node(r+1,c), node(r,c))] = limits_col[r][c]
for r in range(10):
    for c in range(9):
        edge_speedlim[(node(r,c), node(r,c+1))] = edge_speedlim[(node(r,c+1), node(r,c))] = limits_row[r][c]

for u,v in segments:
    s = edge_speedlim[(u,v)]
    c,t = cost_time(s, s)
    base_cost += c
    base_time += t
base_cost, base_time = round(base_cost,4), round(base_time,4)

# ---------- 如果路线一已便宜于 C_min，直接退出 ----------
if base_cost <= C_min + 1e-6:
    print("路线一原本已最便宜，无需优化")
    exit()

budget = C_min - base_cost        # ≤ 0 亦可，DP 会检查
print(f"C_min={C_min:.4f},  baseline={base_cost:.4f},  budget={budget:.4f}")

# ---------- 为多重背包准备选项 ----------
choices=[]
for u,v in segments:
    s_lim = edge_speedlim[(u,v)]
    c0,t0 = cost_time(s_lim, s_lim)
    options=[]
    for pct in (0,20,50,70):
        v_real = s_lim*(1+pct/100)
        c1,t1 = cost_time(v_real, s_lim)
        options.append((round(c1-c0,8), round(t0-t1,8), pct, v_real))
    choices.append(options)

# ---------- DP ----------
scale = 100     # 把元转成分
shift = 300000  # 足够大
states={shift:(0.0,[])}     # key→(time_saved, plan)

for options in choices:
    nxt={}
    for key,(tsave,plan) in states.items():
        for dc,dt,pct,real_v in options:
            nk = key + int(round(dc*scale))
            total = base_cost + (nk-shift)/scale
            if total - C_min > 1e-6:   # 违反约束
                continue
            if nk not in nxt or tsave+dt > nxt[nk][0]+1e-12:
                nxt[nk]=(tsave+dt, plan+[(pct,dc,dt,real_v)])
    states=nxt
    if not states:
        break

if not states:
    print("在费用约束下无可行超速方案，维持限速行驶。")
    best_time, best_cost = base_time, base_cost
    best_plan=[(0,0,0,s) for (_,s) in ((seg,edge_speedlim[seg]) for seg in segments)]
else:
    best_key,max_pair = max(states.items(), key=lambda kv:kv[1][0])
    save_t, best_plan = max_pair
    delta_c = (best_key-shift)/scale
    best_cost = round(base_cost + delta_c,4)
    best_time = round(base_time - save_t,4)

# ---------- 输出 ----------
print("\n===== 结果 =====")
print(f"最短可行时间 : {best_time:.4f} 小时")
print(f"对应总费用   : {best_cost:.4f} 元 (C_min = {C_min:.4f})\n")
print("路线一各段超速方案 (u->v, 限速, 超速%, 实际速度, Δcost, Δtime)")
for (u,v),(pct,dc,dt,real_v) in zip(segments, best_plan):
    s_lim=edge_speedlim[(u,v)]
    print(f"{u+1:3d}->{v+1:<3d} | {s_lim:5g} | {pct:3d}% | {real_v:7.2f} | {dc:+9.4f} | {dt:+9.4f}")
