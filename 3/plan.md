下面我将按顺序给出**第二部分到第五部分**的完整实现思路和示例代码（Python + `heapq`），方便你直接拿去写。假设你已经完成了第一部分，得到了：  
- 边列表 `edges_df`（含 `eid,u,v,vmax`）  
- 时间与费用矩阵 `t_mat[eid, k]`, `c_mat[eid, k]`，其中 `k=0,1,2,3` 对应超速档 `{0,0.2,0.5,0.7}`  

同时你已经拿到问题一的**最短时间路径**（路线一）的节点序列 `route1_nodes = [n0, n1, …, nm]`。
Route One Node Sequence (0-indexed):
0 -> 1 -> 11 -> 12 -> 13 -> 23 -> 24 -> 34 -> 44 -> 54 -> 55 -> 56 -> 57 -> 58 -> 68 -> 78 -> 88 -> 89 -> 99

Route One Node Sequence (1-indexed for reference.md format):
1 -> 2 -> 12 -> 13 -> 14 -> 24 -> 25 -> 35 -> 45 -> 55 -> 56 -> 57 -> 58 -> 59 -> 69 -> 79 -> 89 -> 90 -> 100

---

## 2. 计算竞争对手的最低期望费用 \(C_{\mathrm{bound}}\)

**目标**：假设对每条边都可以任选超速档，则全网最小期望总费用是多少？  
即对每条 `eid`，先取 `min_k c_mat[eid,k]`，再做一次 Dijkstra。

```python
import numpy as np
from heapq import heappush, heappop

# —— 2.1 构造每条边的最小费用
min_edge_cost = np.min(c_mat, axis=1)  # shape = (E,)

# —— 2.2 建邻接表：adj[u] = list of (v, eid)
from collections import defaultdict
adj = defaultdict(list)
for _, row in edges_df.iterrows():
    eid, u, v = int(row.eid), int(row.u), int(row.v)
    adj[u].append((v, eid))

# —— 2.3 Dijkstra over 1~100 节点
INF = 1e18
dist = [INF]*101  # 1-based
start, goal = route1_nodes[0], route1_nodes[-1]
dist[start] = 0
hq = [(0, start)]
while hq:
    d,u = heappop(hq)
    if d>dist[u]: 
        continue
    if u==goal:
        break
    for v, eid in adj[u]:
        nd = d + min_edge_cost[eid]
        if nd < dist[v]:
            dist[v] = nd
            heappush(hq, (nd, v))

C_bound = dist[goal]  # 竞争对手最小期望费用上界
print(f"C_bound = {C_bound:.4f} 元")
```

- **时间复杂度**：\(O(E\log V)\)，秒级可解。  
- **结果**：`C_bound` 即步骤 2 的输出。

---

## 3. 映射路线一到“多重选择背包”问题

**目标**：路线一有 \(m\) 条边，我们要给每条边选一个档 \(k\)，使总费用 \(\sum c_{ek}\le C_{\mathrm{bound}}\)，且总时间最小。

```python
# —— 3.1 从节点序列反推边序列
# 建一个从 (u,v) 到 eid 的映射
uv2eid = {(row.u, row.v): row.eid for _,row in edges_df.iterrows()}

# 把 route1_nodes 转成边列表 route1_eids
route1_eids = []
for u,v in zip(route1_nodes, route1_nodes[1:]):
    eid = uv2eid[(u,v)]
    route1_eids.append(eid)

m = len(route1_eids)
print(f"路线一共有 {m} 条边")

# —— 3.2 为背包准备 weight[i][k] = c_mat[eid, k], value[i][k] = t_mat[eid, k]
# 注意：为了 DP 我们需要把费用放大 10000 并转成整数
MULT = 10000
W = int(np.floor(C_bound * MULT))  # 容量上界，向下取整

# 构造二维列表
weights = [ [ int(round(c_mat[eid,k]*MULT)) for k in range(4) ] 
            for eid in route1_eids ]
values  = [ [ t_mat[eid,k]               for k in range(4) ] 
            for eid in route1_eids ]
```

此时：  
- `weights[i][k]` 是第 \(i\) 条边第 \(k\) 档的“费用”  
- `values[i][k]` 是对应的“时间”  

---

## 4. 动态规划求解 MCKP

我们用 **一维滚动数组** DP，`dp[s]` 表示“总费用恰好为 \(s\) 时能获得的最小总时间”。初始化 `dp[0]=0`，其它为 `+∞`。

```python
# —— 4.1 初始化
dp = [float('inf')] * (W+1)
# back[i][s] 记录第 i 类选档，用于回溯；初始化为 -1
back = [ [-1]*(W+1) for _ in range(m) ]

dp[0] = 0.0

# —— 4.2 分组背包 DP
for i in range(m):
    new_dp = [float('inf')] * (W+1)
    new_back = [ [-1]*(W+1) for _ in range(m) ]  # 只填第 i 行
    for s in range(W+1):
        if dp[s] == float('inf'):
            continue
        # 在第 i 类（第 i 条边）选档 k
        for k in range(4):
            w = weights[i][k]
            v = values[i][k]
            s2 = s + w
            if s2 <= W:
                t_candidate = dp[s] + v
                if t_candidate < new_dp[s2]:
                    new_dp[s2] = t_candidate
                    new_back[i][s2] = k
    dp = new_dp
    # 把第 i 行 back 写回到 back[i]
    for s2,k in enumerate(new_back[i]):
        back[i][s2] = k

# —— 4.3 找到最优解：在所有 s<=W 中找 dp[s] 最小的位置 s*
best_s = min(range(W+1), key=lambda s: dp[s])
best_time = dp[best_s]
best_cost = best_s / MULT

print(f"最短可达时间 = {best_time:.4f} 小时；对应费用 = {best_cost:.4f} 元")
```

- **解释**：  
  - 每次循环 `i`（第 `i` 条边），我们由旧的 `dp` 推出新的 `new_dp`；  
  - `back[i][s2] = k` 表示要在第 `i` 条边选第 `k` 档，才会到达费用 `s2` 的最优状态。  

---

## 5. 回溯超速方案并输出

```python
# —— 5.1 回溯选档
s = best_s
chosen_ks = [0]*m  # 存放每条边的超速档索引
for i in reversed(range(m)):
    k = back[i][s]
    chosen_ks[i] = k
    s -= weights[i][k]

# —— 5.2 输出结果
print("最终超速方案（与路线一边序对应）：")
for idx, (eid, k) in enumerate(zip(route1_eids, chosen_ks), start=1):
    x = BRACKETS[k]
    print(f"  第{idx}段 (eid={eid}) 超速倍率 = {x:.4f}")

print(f"\n汇总：最短可达时间 = {best_time:.4f} 小时")
print(f"     对应期望费用 = {best_cost:.4f} 元")
```

- `chosen_ks[i]` 即第 \(i\) 条边选的档，映射到实际倍率 `BRACKETS[k]`。  
- 最终输出满足题意「在保持路线一为时间最短的前提下，把费用控制到竞争对手最低，再在此基础上尽可能缩短时间」。

---

## 6. （可选）验证

- **重算总时间**：  
  ```python
  total_t = sum(values[i][chosen_ks[i]] for i in range(m))
  assert abs(total_t - best_time) < 1e-6
  ```
- **重算总费用**：  
  ```python
  total_c = sum(weights[i][chosen_ks[i]] for i in range(m)) / MULT
  assert abs(total_c - best_cost) < 1e-6
  ```
- **Monte-Carlo 仿真**（选做，检验期望罚款模型）：  
  - 按 `P_det` 随机判定罚款，跑 1e4 次，统计平均费用与 DP 值对比。  

---

这样，你就有了从“构造竞争上界”到“MCKP DP 选档”再到“结果回溯输出”的完整代码骨架。按此改写、调试，即可得到问题三的最优超速时间和方案。祝编码顺利！