下面给出一套**可直接落地的“确定性受约束最短路径”**求解流程。所有说明均以 Python 实现为例，但核心思想与语言无关；您可以按需改写成 C++、Java 等。

---

## 0. 前置：弄清三件事

| 要素 | 说明 |
| --- | --- |
| **状态维度** | 结点 `i` + 已耗时（离散为若干 time‐slot） |
| **动作** | 选下一条边 `e=(i→j)` + 超速档 \(x\in\{0,0.2,0.5,0.7\}\) |
| **转移代价** | 期望费用 \(c(e,x)\)（含油费、餐饮/住宿/游览费、高速费、期望罚款） |

只要把 **“剩余时间”** 这一维离散化，我们就得到一个普通的加权有向图，随后跑一次 Dijkstra / A\* / Label‐Setting 就能求最优解。

---

## 1. 数据准备

```python
import numpy as np
import pandas as pd

# 1.1 读入 CSV
limits_col = pd.read_csv('limits_col.csv', header=None).values  # 9×10
limits_row = pd.read_csv('limits_row.csv', header=None).values  # 10×9

# 1.2 构造 180 条有向边
EDGE_LEN = 50  # km
edges = []     # [from_id, to_id, v_max]

# 竖向（向上行驶）——节点编号：r*10+c → (r+1)*10+c
for r in range(9):
    for c in range(10):
        n1, n2 = r*10+c+1, (r+1)*10+c+1          # +1 因题目节点从 1 开始
        v = limits_col[r, c]
        edges.append((n1, n2, v))

# 横向（向右行驶）
for r in range(10):
    for c in range(9):
        n1, n2 = r*10+c+1, r*10+(c+1)+1
        v = limits_row[r, c]
        edges.append((n1, n2, v))
```

---

## 2. 预计算 4 档速度的 **时间 & 成本表**

> **一次性算好，后面查询不再重复公式。**

```python
BRACKETS = [0.0, 0.2, 0.5, 0.7]  # 超速倍率
GAS_PRICE = 7.76                 # 元/升
C_FOOD = 20                      # 元/小时
TOLL_RATE = 0.5                  # 高速每公里
EDGE_KM = 50

def fuel_consumption(v):  # 每 100 km 耗油量
    return 0.0625*v + 1.875  # L/100km

def fine_amount(v_max, x):
    ratio = x
    v_class = v_max
    # 分段法规（摘录自题干）
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
fixed_radar = lambda v_max: v_max >= 90
MOBILE_NUM = 20
EDGE_NUM = len(edges)
rho = MOBILE_NUM / EDGE_NUM      # 移动雷达到达某条边的概率（平均化假设）

label_cost = {}  # (edge_id, bracket_idx) -> (time_h, cost)

for eid, (u, v, vmax) in enumerate(edges):
    # 有无雷达
    has_fix = fixed_radar(vmax)
    for k, x in enumerate(BRACKETS):
        v_real = vmax * (1 + x)
        t_hr = EDGE_KM / v_real

        # ——油费
        gas_cost = fuel_consumption(v_real) * EDGE_KM/100 * GAS_PRICE
        # ——餐饮 / 住宿 / 游览
        travel_cost = C_FOOD * t_hr
        # ——高速收费
        toll_cost = (EDGE_KM * TOLL_RATE) if vmax == 120 else 0

        # ——期望罚款
        P_det = 0
        if x > 0:  # 只在超速时才有罚款
            # 先算单个雷达命中率
            p_single = 0.7 if x <= 0.2 else 0.9 if x <= 0.5 else 0.99
            # 雷达出现与否
            num_radars = int(has_fix)  + np.random.binomial(1, rho)  # 期望值=...
            # 期望探测概率
            P_det = 1 - (1 - p_single)**num_radars
        exp_fine = P_det * fine_amount(vmax, x)

        total = gas_cost + travel_cost + toll_cost + exp_fine
        label_cost[(eid, k)] = (t_hr, total)
```

> **说明**  
> - 这里对移动雷达采取“期望一台”的极简模型；如果您有更精细的部署分布，可把 `num_radars` 换成实际分布中的期望值。  
> - 罚款概率是期望值，后续算法最小化的就是**期望总费用**。  

---

## 3. 离散化时间轴

1. **求基准时间 \(T_1\)**：  
   - 跑一次「**所有边按限速行驶**」的最短时间路（可用 Dijkstra，边权=50/v_max），取其总时间即 \(T_1\)。  
2. **设定上限 \(T_\text{max}=0.7 T_1\)**。  
3. **时间分辨率**：常用 1 min(=1/60 h)。定义 `SLOT = 1/60`。  
4. **最大槽数**：`N_T = ceil(T_max / SLOT)`。

---

## 4. 构造 *扩展图* \( (node, t)\) → (next_node, t+Δt) **Label-Setting 最短路**

```python
from heapq import heappush, heappop
T_max = 0.7 * T1    # ← 第 3 步求得
SLOT = 1/60         # 1 分钟
N_T = int(np.ceil(T_max / SLOT))

INF = 1e18
# dist[i][s] = 到状态 (节点 i, 第 s 个时间槽) 的最小费用
dist = [ [INF]* (N_T+1) for _ in range(101) ]  # 节点 1~100
prev = {}  # (i,s) -> (prev_i, prev_s, edge_id, bracket)

start, goal = 1, 100
dist[start][0]=0
hq=[(0, start, 0)]  # (cost, node, slot)

while hq:
    c,i,s = heappop(hq)
    if (i==goal):   # 已在可行时间内到终点，且堆顶保证最小
        break
    if c>dist[i][s]: continue
    # 遍历以 i 为起点的所有出边
    for eid,(u,v,vmax) in enumerate(edges):
        if u!=i: continue
        for k,x in enumerate(BRACKETS):
            t_inc , cost_inc = label_cost[(eid,k)]
            s2 = s + int(np.ceil(t_inc/SLOT))
            if s2>N_T: continue    # 超时，丢弃
            new_cost = c + cost_inc
            if new_cost < dist[v][s2]:
                dist[v][s2]=new_cost
                prev[(v,s2)]=(i,s,eid,k)
                heappush(hq,(new_cost,v,s2))
```

> **细节要点**  
> 1. **扩展图节点 ≤ 100×(N_T+1)**，N_T 一般在 300～600（如果 T_1≈8 h，则 \(0.7T_1≈5.6 h\)，用 1 min 离散就是 336 个槽）。总节点数 ≈3.4×10⁴，可轻松秒级求解。  
> 2. 使用 **堆优化 Dijkstra**（或 A\*，将剩余曼哈顿距离 / vmax 作为启发式）可再提速。  
> 3. 若想进一步减小状态，可把时间槽拉大到 2 min、5 min；不影响可行性但分辨率变粗。  

---

## 5. 回溯得到 **路线三 + 超速方案**

```python
# 5.1 找终点最小费用的 (goal, s*)
s_goal = min(range(N_T+1), key=lambda s: dist[goal][s])
min_cost = dist[goal][s_goal]

# 5.2 回溯路径
path_edges, speed_brackets = [], []
node, slot = goal, s_goal
while (node,slot)!=(start,0):
    pre_i, pre_s, eid, k = prev[(node,slot)]
    path_edges.append(eid)
    speed_brackets.append(k)
    node,slot = pre_i, pre_s

# 5.3 还原节点序列（逆序→正序）
route_nodes=[start]
for eid in reversed(path_edges):
    route_nodes.append(edges[eid][1])

route_nodes      # 路口编号序列
speed_brackets[::-1]  # 同序长度  len(route_nodes)-1
```

> 输出示例（假设）：  
> ```
> 路线三：1, 2, 3, …, 100
> 超速方案（与上述顺序对应）：
>   边 1: x=0.2
>   边 2: x=0.0
>   ...
> ```  
> 超速倍率可在最终表里直接给出 0, 0.2, 0.5, 0.7；如题干要求 4 位小数则输出 0.2000 等。

---

## 6. 验证与期望成本

1. **计算总时间**（再次用精确 `t_inc` 累加）检查确实 ≤ 0.7 T₁。  
2. **Monte-Carlo 仿真**（可选）  
   - 对每段按罚款概率随机决定是否被抓，累和罚款。  
   - 1 万轮可得期望 ± 几元的置信区间，验证 DP 结果合理性。  

---

## 7. 可能的改进 / 变体

| 方向 | 说明 |
| --- | --- |
| **更细档位** | 把超速倍率改为 0.05 步长，方法不变但状态×7；依然秒至十几秒可解。 |
| **软约束** | 无需离散时间，直接把 \(λ\bigl(T-0.7T_1\bigr)^+\) 加进目标；调 λ 用二分搜索直到满足时间要求。 |
| **多目标 Pareto** | 同时输出“最省钱 / 次省钱 / …”多条可行路线，可在 Label‐Setting 中为同一点保留前 k 个互斥支配标签即可。 |

---

## 8. 小结

> **核心一句话**：  
> **把“剩余时间”离散化后，每个状态就是普通图节点，求“期望费用”最短路即可。**  

- 实现量 ≈ 200 行 Python（含 I/O & 回溯）。  
- 不依赖第三方优化库，只用 `heapq` 即可。  
- 算法 **确定可解释**，最差复杂度 \(O(|V| \log|V| + |E||B|)\)，这里 |V|≈3×10⁴、|E||B|≈7×10³。  
- 若未来要加入“路段临时限速”“交通管制”或“移动雷达真实分布”，只需改 **label_cost** 即可，无须改动核心求解器。  

照此流程即可稳妥算出题目要求的 **路线三** 及其超速策略，并保证总期望费用最小且满足时间上限。祝您编码顺利!