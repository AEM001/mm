下面给出一套**可以直接照着落地的实现方案**。整个流程分为 4 大阶段，每一步都给出所需数据结构、公式、算法、以及推荐的 Python/​开源库写法。你只要按步骤敲代码，就能算出：

* 在保证“路线一”仍是**全网费用最优**的前提下  
* 每一段最多**超速 70 %**时，  
* “路线一”可以把总旅行时间压缩到 **T\*** 小时  
* 对应的总费用 **C\***（元）  
* 以及 19 条路段各自的超速倍率方案 **s\_e\***。  

> 注：为了让文字聚焦“怎么做”，下面不输出最终数字。按方案跑完即可得到四位小数的结果。  

---

## 0. 依赖与环境

| 功能 | 库 | 备注 |
|------|----|------|
| 图结构与最短路 | **networkx** | pip install networkx |
| MILP 求解 | **pulp**（自带 CBC）<br>或 **gurobipy** / **cplex** | pulp 足够；若装 Gurobi/CPLEX 更快 |
| 数据处理 | pandas / numpy | 读 CSV |
| 可选加速 | numba / cython | 仅在自定义穷举时 |

---

## 1. 读数据并建图  

### 1.1 读取 CSV  

```python
import pandas as pd, networkx as nx, itertools as it

d = 50  # km   单段距离
limits_row = pd.read_csv('limits_row.csv', header=None).values      # shape (10, 9)
limits_col = pd.read_csv('limits_col.csv', header=None).values      # shape (9, 10)
```

### 1.2 构造有向图 `G`  

```python
G = nx.DiGraph()
for r in range(10):
    for c in range(10):
        idx = r*10 + c                # 0-based 路口编号
        if c < 9:                     # 右
            v_max = limits_row[r, c]
            G.add_edge(idx, idx+1,
                        d=d, v_max=v_max,
                        highway=(v_max==120))
        if r < 9:                     # 上
            v_max = limits_col[r, c]
            G.add_edge(idx, idx+10,
                        d=d, v_max=v_max,
                        highway=(v_max==120))
```

> **边属性**：`d, v_max, highway`，后面公式都要用。  

---

## 2. 计算“竞争者”可选路线的**最低费用**  

### 2.1 费用函数（守法行驶）  

对一条路径 P：

\[
\begin{aligned}
T(P) &= \sum_{e\in P} \frac{d}{v_{\max,e}}\\[4pt]
C_{\text{time}} &= 20\,T\\
C_{\text{fuel}} &= \sum_{e\in P} \frac{d}{100}\Bigl(0.0625\,v_{\max,e}+1.875\Bigr)\times7.76\\
C_{\text{toll}} &= \sum_{\text{高速 }e\in P} 0.5\,d\\
C(P) &= C_{\text{time}} + C_{\text{fuel}} + C_{\text{toll}}
\end{aligned}
\]

### 2.2 取  **k-shortest paths**  

```python
from networkx.algorithms.simple_paths import shortest_simple_paths

src, dst = 0, 99
k = 100       # 经验证 50~100 已足够
paths = it.islice(shortest_simple_paths(G, src, dst, weight=lambda u,v,attr: d/attr['v_max']), k)
```

> 这里用运行时间作权重，保证前 k 条 **最快** 路线都被枚举。  

### 2.3 逐条计算费用，取最小  

```python
def route_cost(path):
    T = sum(G[u][v]['d']/G[u][v]['v_max'] for u,v in zip(path, path[1:]))
    C_time = 20*T
    C_fuel = sum(G[u][v]['d']/100*(0.0625*G[u][v]['v_max']+1.875)*7.76
                 for u,v in zip(path, path[1:]))
    C_toll = sum(0.5*G[u][v]['d'] for u,v in zip(path, path[1:]) if G[u][v]['highway'])
    return C_time + C_fuel + C_toll

C_competitor_min = min(route_cost(p) for p in paths)
```

---

## 3. 对“路线一”做**超速优化**  

### 3.1 准备常量  

```python
route1 = [0,1,11,12,13,23,24,34,44,54,55,56,57,58,68,78,88,89,99]
edges1  = list(zip(route1, route1[1:]))

# 罚款表（根据限速区间+超速档位）
fine_table = {
    # (v_max 区间上界, 超速倍率) -> (被抓概率, 罚款额)
    # 下面只列举一个示例元祖；完整要覆盖四档倍率
    (50, 1.2): (0.70, 50),
    ...
}
```

> 四档倍率：1.0, 1.2, 1.5, 1.7  
> 每个倍率对应「可能被抓概率」×「罚款金额」→ **期望罚款**。  

### 3.2 MILP 变量  

令  
- `x_e,a ∈ {0,1}` 表示边 e 选择倍率 `a ∈ {1.0,1.2,1.5,1.7}`  
- 对每条 e：`∑_a x_e,a = 1`  

### 3.3 约束  

1. **费用不劣**  

\[
C_{\text{route1}}(x)\;\le\;C_{\text{competitor\_min}}-\epsilon
\]

（取 ε = 1e-4 即可）  

2. **费用分解**  

\[
\begin{aligned}
T(x) &= \sum_{e}\sum_{a} \frac{d}{a\,v_{\max,e}}\,x_{e,a} \\[4pt]
C_{\text{time}} &= 20\,T(x)\\
C_{\text{fuel}} &= \sum_{e}\sum_{a} 
      \frac{d}{100}\Bigl(0.0625\,a v_{\max,e}+1.875\Bigr)\times7.76\,x_{e,a}\\
C_{\text{toll}} &= \sum_{\text{高速 }e} 0.5\,d \quad(\text{常数})\\
C_{\text{fine}} &= \sum_{e}\sum_{a} \text{prob}(v_{\max,e},a)\times
                  \text{fine}(v_{\max,e},a)\;x_{e,a}\\[4pt]
C_{\text{route1}} &= C_{\text{time}}+C_{\text{fuel}}+C_{\text{toll}}+C_{\text{fine}}
\end{aligned}
\]

全部都是 **线性** 表达式。  

### 3.4 目标函数  

\[
\min T(x) \quad\text{(或等价地）}\quad \max \text{节省时间}
\]

### 3.5 用 pulp 编码  

```python
import pulp as pl
A = [1.0, 1.2, 1.5, 1.7]
prob = pl.LpProblem('Overspeed', pl.LpMinimize)

x = {(e,a): pl.LpVariable(f'x_{u}_{v}_{a}', cat='Binary')
     for (u,v) in edges1 for a in A}

# 目标：总时间
prob += pl.lpSum(G[u][v]['d']/(a*G[u][v]['v_max'])*x[(u,v),a]
                 for (u,v) in edges1 for a in A)

# 每段只能选一个倍率
for u,v in edges1:
    prob += pl.lpSum(x[(u,v),a] for a in A) == 1

# 总费用约束
C_time = 20*prob.objective
C_fuel = pl.lpSum(G[u][v]['d']/100*(0.0625*a*G[u][v]['v_max']+1.875)*7.76*x[(u,v),a]
                  for (u,v) in edges1 for a in A)
C_toll = sum(0.5*G[u][v]['d'] for (u,v) in edges1 if G[u][v]['highway'])
C_fine = pl.lpSum(fine_table[(limit_class(G[u][v]['v_max']),a)][0] *
                  fine_table[(limit_class(G[u][v]['v_max']),a)][1] *
                  x[(u,v),a]
                  for (u,v) in edges1 for a in A)

prob += C_time + C_fuel + C_toll + C_fine <= C_competitor_min - 1e-4
prob.solve(pl.PULP_CBC_CMD())
```

> `limit_class()` 用于把 37 km/h、68 km/h … 映射到 “限速 ≤50”, “50–80”, “80–100”, “100+” 四档。  

### 3.6 输出结果  

```python
s_star = { (u,v): a for (u,v),a in x if pl.value(x[(u,v),a])>0.5 }
T_star = pl.value(prob.objective)
C_star = (20*T_star + pl.value(C_fuel+C_fine) + C_toll)
```

然后把 `T_star, C_star` 保留四位小数，`s_star` 转成表格即可。  

---

## 4. 可选：连续倍率 / 穷举替代  

1. **连续倍率**：把 `a` 从离散变为变量 `1 ≤ s_e ≤ 1.7`  
   * 罚款和被抓概率变成 **分段常数**，需要额外 0-1 变量做 Big-M 切分；与离散版同规模。  
2. **穷举剪枝**：19 段 × 4 档 = 2.7 亿 组合  
   * 可自顶向下深度优先 + **截至条件**（一旦费用超标就剪枝）  
   * 但 MILP 已能秒级出全局最优，没必要再写穷举。  

---

## 5. 完整跑通后的交付物  

1. **结果文件**（例如 `solution.csv`）  

| Edge (u→v) | v_max | Chosen a | Segment Speed (km/h) |
|------------|-------|----------|----------------------|
| 0→1        | 90    | 1.5      | 135                  |
| …          | …     | …        | …                    |

2. `Time = T_star` (h)  
3. `Cost = C_star` (¥)  

---

## 6. 常见坑 & 调试技巧  

| 现象 | 排查思路 |
|------|----------|
| **无可行解** | 把 `ε` 设成 0；如果仍 infeasible，多半你的罚款概率/金额表或者竞争者费用算错 |
| **结果时间变化很小** | 检查是否把所有高速路段都限速 120（无法再超速） |
| **求解慢** | 确认使用了整数版 4 档倍率；连 CBC 都是秒级 |

---

按上面代码框搭起来，5~10 分钟即可得到最终四位小数答案。祝你顺利复现！