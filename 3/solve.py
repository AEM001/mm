# -*- coding: utf-8 -*-
"""
求解“路线一”在超速(<=70%)情况下保持最优费用时, 可压缩的最短时间及对应费用。
步骤:
1. 读入 limits_col.csv / limits_row.csv
2. 构建 10×10 有向图(100 节点, 180 条边)
3. 用 k‑shortest paths + 离散超速档 {1.0,1.2,1.5,1.7} 计算竞争者能做到的最低费用
4. 在固定的路线一 19 条边上建 MILP(变量=19×4 one‑hot),
   目标最小总时间, 约束总费用 ≤ 竞争者最小费用‑ε
5. 输出最短时间、对应费用、各边超速倍率
依赖: pandas, numpy, networkx, pulp
安装: pip install pandas numpy networkx pulp
"""

import pandas as pd
import numpy as np
import networkx as nx
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, lpSum, value, PULP_CBC_CMD

############################################################
# 1. 读入限速矩阵 (请保证 csv 位于脚本同级目录)
############################################################
limits_col = pd.read_csv('D:\Code\mm\data\limits_col.csv', header=None).values   # 9×10, (r,c)->(r+1,c)
limits_row = pd.read_csv('D:\Code\mm\data\limits_row.csv', header=None).values   # 10×9, (r,c)->(r,c+1)

############################################################
# 2. 构建路网
############################################################
d = 50  # km, 相邻节点距离
G = nx.DiGraph()

def node_id(r: int, c: int) -> int:
    """将 (行,列) 映射到 0‑99 的节点编号(0‑index)"""
    return r * 10 + c

# 纵向边
for r in range(9):
    for c in range(10):
        v_max = limits_col[r, c]
        G.add_edge(node_id(r, c), node_id(r + 1, c), v_max=v_max, highway=(v_max == 120))

# 横向边
for r in range(10):
    for c in range(9):
        v_max = limits_row[r, c]
        G.add_edge(node_id(r, c), node_id(r, c + 1), v_max=v_max, highway=(v_max == 120))

############################################################
# 3. 费用模型辅助函数
############################################################
# 超速档以及对应抓拍概率
BETA_LEVELS = [1.0, 1.2, 1.5, 1.7]
DETECTION_PROB = {1.0: 0.0, 1.2: 0.70, 1.5: 0.90, 1.7: 0.99}

def detection_prob(beta: float) -> float:
    return DETECTION_PROB[beta]

# 罚款金额查表(单位: 元)
# key=(限速区间上界, beta)
_FINE_TABLE = {
    # 限速 <50 km/h
    (50, 1.2): 50,  (50, 1.5): 100, (50, 1.7): 300,
    # 50‑80
    (80, 1.2): 100, (80, 1.5): 150, (80, 1.7): 500,
    # 80‑100
    (100,1.2): 150, (100,1.5): 200, (100,1.7): 1000,
    # >100
    (float('inf'), 1.2): 0,   # 未用(<=50% 无罚款要求, 但题意 >100 只给 50% 档起罚)
    (float('inf'), 1.5): 200, (float('inf'),1.7): 1500,
}

def fine_amount(v_max: int, beta: float) -> float:
    """返回超速倍率 beta 下的罚款金额 (不含检测概率)"""
    if beta <= 1.0 or beta <= 1.2:
        return 0.0
    # 查限速区间
    limit_keys = sorted([k for k, _ in _FINE_TABLE])
    for lim in limit_keys:
        if v_max < lim:
            return _FINE_TABLE[(lim, beta)]
    # v_max>=100
    return _FINE_TABLE[(float('inf'), beta)]

# 单边费用组成
def edge_cost(v_max: int, is_hw: bool, beta: float):
    """返回 (时间费, 油费, 罚款费用)"""
    # 时间、油耗
    t = d / (beta * v_max)  # h
    C_time = 20 * t
    liters = d * (0.0625 * beta * v_max + 1.875) / 100
    C_fuel = 7.76 * liters
    # 罚款(期望)
    C_fine = detection_prob(beta) * fine_amount(v_max, beta)
    # 高速费单独算
    return C_time, C_fuel, C_fine

############################################################
# 4. 竞争者最低费用 (k‑shortest paths × overspeed 枚举)
############################################################
from itertools import islice, product

START, END = 0, 99
k_paths_gen = nx.shortest_simple_paths(G, START, END)  # 默认按 hop 数排序
k = 50  # 可调
paths = list(islice(k_paths_gen, k))

competitor_cost = float('inf')
for path in paths:
    for beta in BETA_LEVELS:  # 把同档统一超速给竞争者, 已够保守
        C_time = C_fuel = C_fine = 0
        C_toll = 0
        for u, v in zip(path[:-1], path[1:]):
            e = G[u][v]
            v_max, is_hw = e['v_max'], e['highway']
            ct, cf, cfi = edge_cost(v_max, is_hw, beta)
            C_time += ct
            C_fuel += cf
            C_fine += cfi
            if is_hw:
                C_toll += 0.5 * d
        total = C_time + C_fuel + C_fine + C_toll
        competitor_cost = min(competitor_cost, total)

print(f"竞争者最低费用 ≈ {competitor_cost:.2f} 元")

############################################################
# 5. MILP 建模 (固定路线一)
############################################################
route1 = [0,1,11,12,13,23,24,34,44,54,55,56,57,58,68,78,88,89,99]
edges = list(zip(route1[:-1], route1[1:]))

model = LpProblem("Route1_Speed_Opt", LpMinimize)
# one‑hot 变量 x[i,l]
x = {(i, l): LpVariable(f"x_{i}_{l}", cat=LpInteger, lowBound=0, upBound=1)
     for i in range(len(edges)) for l in range(len(BETA_LEVELS))}

# 每条边恰选 1 档
for i in range(len(edges)):
    model += lpSum(x[i, l] for l in range(len(BETA_LEVELS))) == 1

# 目标:总时间
objective_time = lpSum(
    d / (BETA_LEVELS[l] * G[u][v]['v_max']) * x[i, l]
    for i, (u, v) in enumerate(edges)
    for l in range(len(BETA_LEVELS))
)
model += objective_time

# 费用约束
cost_expr = 0
for i, (u, v) in enumerate(edges):
    v_max, is_hw = G[u][v]['v_max'], G[u][v]['highway']
    for l, beta in enumerate(BETA_LEVELS):
        ct, cf, cfi = edge_cost(v_max, is_hw, beta)
        cost_expr += (ct + cf + cfi) * x[i, l]
        if is_hw:
            cost_expr += 0.5 * d * x[i, l]
# 再加一次常数高速费(与档位无关)以免遗漏
cost_expr += 0.5 * d * sum(1 for u, v in edges if G[u][v]['highway'])

EPS = 1e-3
model += cost_expr <= competitor_cost - EPS, "CostCap"

############################################################
# 6. 求解 & 输出结果
############################################################
solver = PULP_CBC_CMD(msg=False)
model.solve(solver=solver)

best_time = value(objective_time)
best_cost = value(cost_expr)

beta_selected = [
    sum(BETA_LEVELS[l] * value(x[i, l]) for l in range(len(BETA_LEVELS)))
    for i in range(len(edges))
]

print("\n******** 最优结果 ********")
print(f"最短行驶时间: {best_time:.4f} 小时")
print(f"对应总费用:   {best_cost:.4f} 元\n")
print("各路段超速倍率:")
for (u, v), b in zip(edges, beta_selected):
    print(f"{u:2d} -> {v:2d} : ×{b:.1f}")

############################################################
# main 入口(可选)
############################################################
if __name__ == "__main__":
    pass  # 直接在 import 时已执行, 保留给需要封装为函数/模块时使用
