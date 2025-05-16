下面给出一套 **可直接落地的 Primal‑Dual Actor‑Critic（拉格朗日 Actor‑Critic）方案**，把“0.7 T₁ 时限 + 期望费用最小”作为 **CMDP（约束 MDP）** 求解。  
我按“建模 → 网络与输入 → 损失函数与 λ 更新 → 训练流程 → 调参要点 → 路线抽取”六步展开；不写整段代码，而是把关键接口、张量形状、超参数、训练节奏全部说明清楚，方便你用 PyTorch 或 TensorFlow / RLlib / SB3 自行实现。

---

## 1  环境接口（保持与前两题一致，只增添返回量）

| `step(action)` 额外返回 | 说明 |
|-------------------------|------|
| `cost_edge`             | 当前路段旅途+期望罚款（元） |
| `time_edge`             | 当前路段行驶时间（秒，**统一用秒**） |

> **总时间上限** `T_dead = 0.7 T₁`（秒）。 只要 `t_used > T_dead`，环境立即 `done=True` 并再给一次性 **惩罚 -C_big**（可取 –10 000 元），同时返回 `constraint_violation = t_used - T_dead`，否则 0。  
> 这样梯度里仍能感知“超了多少秒”，而不是简单 0/1。

---

## 2  状态与动作表征

### 2.1  状态向量 `s`  (shape = 102)

| 分量 | 维度 | 归一化 |
|------|------|--------|
| 节点编号 one‑hot | 100 | — |
| 剩余时间 `τ = (T_dead - t_used) / T_dead` | 1 | ∈[0, 1] |
| 是否已经超时 flag | 1 | 0/1 |

> **为什么不用 100+ 段的嵌入？** 网络极小，one‑hot 足以；若想压缩可加 `nn.Embedding(100, 8)` 替换 one‑hot，再把 8 维嵌入与剩余时间拼起来送入 MLP。

### 2.2  动作空间

- 离散 16 维：`(dir ∈0‑3,  r_idx ∈0‑3)`  
- 用 **categorical policy** 输出长度 16 的概率向量 π(a∣s)。

---

## 3  网络结构

```
Shared  →  Actor head         (16 logits)
        ↘  Critic(V) head     (1 value)
```

| 层 | 宽度 | 激活 |
|----|------|------|
| Linear | 128 | ReLU |
| Linear | 128 | ReLU |

- **独立参数**：Actor 与 Critic 各有一条分支。  
- **初始化**：`orthogonal(init_gain=√2)`；最后一层 Actor logits设 gain 0.01 便于探索。  
- **输出**：Critic 给 `V(s)` 估计 **Lagrangian 回报**（见下一节）。

---

## 4  损失函数与 λ（拉格朗日乘子）更新

### 4.1  Lagrangian 累积回报  
对每条轨迹收集  
\[
G^R_t   = \sum_{k\ge0} \gamma^k (-\,\mathtt{cost}_{t+k})
,\quad
G^T_t   = \sum_{k\ge0} \gamma^k (\mathtt{time}_{t+k})
\]
其中 γ=0.99,  `cost` 已≥0, `time` 单位秒。  

> 建议用 **GAE(λ=0.95)** 来估计优势；时间量也可用折扣累计，保持公式对称。

### 4.2  Actor 损失  
\[
\mathcal{L}_\pi
=
-\;\mathbb{E}\bigl[
    A^{\text{λ}}_t \; \log \pi_\theta(a_t∣s_t)
    + α_{\text{ent}}\,\mathcal{H}(\pi_\theta(\cdot∣s_t))
  \bigr]
\]
其中  
\[
A^{\text{λ}}_t
=
G^R_t
\;-\;
\lambda \, G^T_t
\;-\;
V_\phi(s_t)
\]

- `α_ent = 0.01` —— 熵正则，防止策略骤然收敛到局部最优。  
- **注意**：Critic学习的目标是  
  \[
  V_\phi(s_t) ≈ G^R_t - \lambda\,G^T_t
  \]

### 4.3  Critic损失  
\[
\mathcal{L}_V = \frac12\,(V_\phi(s_t) - (G^R_t - \lambda\,G^T_t))^2
\]

### 4.4  λ 更新（轨迹级、投影到 ≥0）  
采样一批完整回合后  
\[
\lambda \leftarrow
\bigl[
  \lambda + \beta_\lambda \; \bigl(\;\overline{T_{\text{traj}}} - T_{dead}\bigr)
\bigr]^+
\]
- `β_λ = 5 e‑4` 起步；训练到 1e4 步后把 β_λ 乘以 0.3。  
- 若使用 **Softplus λ = softplus(λ_raw)**，则可直接对 λ_raw 施梯度，不用投影。

---

## 5  训练节奏

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| **回合长度** | 最多 200 步 | 到终点或超时即截断 |
| **采样批量** | 2560 步（≈ 32 episode） | 单批进行一次梯度更新 |
| **优化迭代** | 每批 Actor/Critic 各更新 1 次 | 如 PPO 中的 minibatch 若想更稳可 4 次 |
| **学习率** | 1 e‑3 | Adam；Critic 可 5 e‑3 |
| **梯度裁剪** | 5 | 避免爆梯度 |
| **训练总步数** | 150 k 左右即可收敛 | ≈ 5 min/GPU,  <20 min CPU |
| **ε** | 不再用；策略本身随机 |

### Warm‑up 阶段（前 20 k 步）

- 固定 `λ = 0`  
- 只最小化成本，让 Actor 找到可行路线  
- **冻结 Critic** 10 k 步后再同步训练，提高 early 估值稳定度

---

## 6  路线三与超速方案抽取

1. 把 `λ` 固定为收敛时的均值  
2. **贪婪解码**：在环境副本中每步取 `argmax π(a∣s)`  
3. 记录 `node_seq`, `r_percent_seq`, `cost_edge`, `time_edge`  
4. 计算  
   - `T3 = Σ time_edge (h)`  
   - `C_travel = Σ (旅途成本)`  
   - `E_fine = Σ (期望罚款)`  
   - **C2 = C_travel + E_fine**  
5. 若 `T3 > 0.7 T1` 重复第 2 步但把 softmax 温度降到 0.5 再采样，直至满足时限  

---

## 7  调参指北（遇到收敛慢 / λ 爆涨逐项排查）

| 顺序 | 调节 | 观察指标 | 典型目标 |
|------|------|----------|----------|
| ① | `TIME_SCALE` | λ 稳定落在 50–500 | λ 太大→放大 TIME_SCALE |
| ② | `β_λ` | 约束违反均值 → 0 ±0.05h | β_λ 太大→λ振荡 |
| ③ | 熵系数 `α_ent` | 成功率 >90 % | 太小易早收敛 → 提高 |
| ④ | 批量步数 | A²、V 方差 | 批量太小波动大 |
| ⑤ | 网络宽度128→256 | –C₂ 收敛值 | 复杂策略学不动时增宽 |

---

### 核心要点回顾

1. **状态向量**：节点 one‑hot + 归一化剩余时间，完全避免离散时间桶。  
2. **Actor‑Critic**：优势函数里直接减 `λ·G^T`，Critic 拟合同一个 Lagrangian 回报即可。  
3. **λ 更新**：渐进式 dual ascent，批量平均超时量做梯度。  
4. **Warm‑up λ=0** 能让策略先学“怎样走到终点”，再精调时限。  
5. **Softplus λ** + **时间秒级缩放** 是抑制 λ 爆炸的保险丝。  

按此配置，你应能在 <20 万环境步内把成功率推至 98 % 以上，`C₂` 收敛至 2 000–3 000 元量级（视限速图而定），并输出可靠的 **路线三 + 各段超速比例**。祝你实验顺利!