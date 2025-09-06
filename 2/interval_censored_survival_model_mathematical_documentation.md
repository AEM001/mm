# 区间删失生存模型在NIPT最佳时点选择中的数学建模详细过程

## 1. 问题背景与数学框架

### 1.1 问题定义
本研究旨在基于孕妇BMI建立区间删失生存分析模型，确定NIPT（无创产前检测）的最佳检测时点。核心问题是：给定孕妇的BMI和年龄，何时进行NIPT检测能够达到预期的成功率（如90%）。

### 1.2 数学框架
设 $T$ 为Y染色体浓度达到4%阈值的时间（孕周），$X = (X_1, X_2)$ 为协变量向量，其中 $X_1$ 为BMI，$X_2$ 为年龄。我们的目标是建立条件生存函数：

$$S(t|X) = P(T > t | X_1, X_2)$$

其中 $S(t|X)$ 表示在给定BMI和年龄条件下，Y染色体浓度在时间 $t$ 之后才达标的概率。

## 2. 区间删失数据构建

### 2.1 删失类型定义
对于每个孕妇 $i$，设其检测时间序列为 $t_{i1} < t_{i2} < \cdots < t_{ik_i}$，对应的Y染色体浓度为 $y_{i1}, y_{i2}, \ldots, y_{ik_i}$。

定义阈值 $\tau = \text{logit}(0.04) = \log\left(\frac{0.04}{1-0.04}\right) = \log\left(\frac{0.04}{0.96}\right)$

根据达标情况，将数据分为三类：

1. **左删失**：如果 $y_{i1} \geq \tau$，则 $T_i \leq t_{i1}$，观测区间为 $(0, t_{i1}]$
2. **区间删失**：如果存在 $j$ 使得 $y_{ij} < \tau$ 且 $y_{i,j+1} \geq \tau$，则 $t_{ij} < T_i \leq t_{i,j+1}$，观测区间为 $(t_{ij}, t_{i,j+1}]$
3. **右删失**：如果对所有 $j$ 都有 $y_{ij} < \tau$，则 $T_i > t_{ik_i}$，观测区间为 $(t_{ik_i}, +\infty)$

### 2.2 数据标准化
为了数值稳定性，对时间和协变量进行标准化：

$$T_{\text{std}} = \frac{T - \mu_T}{\sigma_T}, \quad X_{j,\text{std}} = \frac{X_j - \mu_{X_j}}{\sigma_{X_j}}$$

其中 $\mu_T = 16.846$，$\sigma_T = 4.076$ 为时间的均值和标准差。

## 3. 加速失效时间(AFT)模型

### 3.1 模型基本形式
AFT模型假设协变量通过加速或减缓失效过程来影响生存时间。对数线性形式为：

$$\log T = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \sigma \varepsilon$$

其中：
- $\beta_0$ 为截距参数
- $\beta_1, \beta_2$ 为BMI和年龄的回归系数
- $\sigma > 0$ 为尺度参数
- $\varepsilon$ 为误差项，其分布决定了生存时间的分布族

### 3.2 Weibull AFT模型
当 $\varepsilon$ 服从极值分布时，$T$ 服从Weibull分布。

#### 3.2.1 概率密度函数
$$f(t|X) = \frac{1}{\sigma t} \phi\left(\frac{\log t - \mu(X)}{\sigma}\right) \exp\left(-\exp\left(-\frac{\log t - \mu(X)}{\sigma}\right)\right)$$

其中 $\mu(X) = \beta_0 + \beta_1 X_1 + \beta_2 X_2$，$\phi(\cdot)$ 为标准极值分布的密度函数。

#### 3.2.2 生存函数
$$S(t|X) = \exp\left(-\exp\left(-\frac{\log t - \mu(X)}{\sigma}\right)\right)$$

#### 3.2.3 对数似然函数
对于观测到的区间删失数据，对数似然函数为：

$$\ell(\boldsymbol{\beta}, \sigma) = \sum_{i=1}^n \log[S(L_i|X_i) - S(R_i|X_i)]$$

其中 $(L_i, R_i]$ 为第 $i$ 个观测的删失区间。

在代码实现中，为了数值稳定性，使用标准化残差：
$$z_i = \frac{\log t_i - \mu(X_i)}{\sigma}$$

对数似然简化为：
$$\ell = \sum_{i=1}^n \left(-z_i - e^{-z_i} - \log \sigma\right)$$

### 3.3 对数正态AFT模型
当 $\varepsilon \sim N(0,1)$ 时，$T$ 服从对数正态分布。

#### 3.3.1 生存函数
$$S(t|X) = 1 - \Phi\left(\frac{\log t - \mu(X)}{\sigma}\right)$$

其中 $\Phi(\cdot)$ 为标准正态分布的累积分布函数。

#### 3.3.2 对数似然函数
$$\ell(\boldsymbol{\beta}, \sigma) = \sum_{i=1}^n \left(-\frac{1}{2}z_i^2 - \frac{1}{2}\log(2\pi) - \log \sigma\right)$$

其中 $z_i = \frac{\log t_i - \mu(X_i)}{\sigma}$。

### 3.4 参数估计
使用最大似然估计法，通过数值优化求解：

$$\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta})$$

其中 $\boldsymbol{\theta} = (\beta_0, \beta_1, \beta_2, \sigma)$。

代码中使用多种优化算法（L-BFGS-B、SLSQP、TNC）以确保收敛到全局最优解。

## 4. BMI分组聚类

### 4.1 K-means聚类
使用K-means算法对BMI进行聚类分组：

$$\min_{C_1,\ldots,C_k} \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2$$

其中 $C_i$ 为第 $i$ 个聚类，$\mu_i$ 为聚类中心。

### 4.2 最优聚类数确定
使用肘部法则确定最优聚类数。计算不同 $k$ 值下的组内平方和（WCSS）：

$$\text{WCSS}(k) = \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2$$

选择使得 $\frac{\text{WCSS}(k-1) - \text{WCSS}(k)}{\text{WCSS}(k) - \text{WCSS}(k+1)}$ 最大的 $k$ 值。

## 5. 最佳时点确定

### 5.1 成功率函数
对于BMI组 $g$，定义成功率函数：

$$F_g(t) = 1 - S(t|\bar{X}_g) = P(T \leq t | \bar{X}_g)$$

其中 $\bar{X}_g = (\bar{X}_{1g}, \bar{X}_{2g})$ 为第 $g$ 组的BMI和年龄均值。

### 5.2 最佳时点计算
给定目标成功率 $\alpha$（如90%），最佳时点定义为：

$$t_g^* = \inf\{t : F_g(t) \geq \alpha\}$$

即首次达到目标成功率的时间点。

### 5.3 风险等级评估
基于最佳时点进行风险分层：

$$\text{风险等级} = \begin{cases}
\text{低风险} & \text{if } t_g^* \leq 12 \text{周} \\
\text{中风险} & \text{if } 12 < t_g^* \leq 27 \text{周} \\
\text{高风险} & \text{if } t_g^* > 27 \text{周}
\end{cases}$$

## 6. 模型预测与应用

### 6.1 个体预测
对于新的孕妇，给定其BMI $x_1$ 和年龄 $x_2$，预测其在时间 $t$ 的成功概率：

$$\hat{F}(t|x_1, x_2) = 1 - \hat{S}(t|x_1, x_2)$$

其中 $\hat{S}(t|x_1, x_2)$ 使用拟合的AFT模型计算。

### 6.2 时间标准化转换
由于模型在标准化时间尺度上拟合，需要进行时间转换：

**标准化到原始时间：**
$$t_{\text{original}} = t_{\text{std}} \times \sigma_T + \mu_T$$

**原始到标准化时间：**
$$t_{\text{std}} = \frac{t_{\text{original}} - \mu_T}{\sigma_T}$$

## 7. 模型验证与诊断

### 7.1 似然比检验
比较不同分布假设下的模型拟合效果：

$$\text{LR} = -2[\ell(\hat{\boldsymbol{\theta}}_0) - \ell(\hat{\boldsymbol{\theta}}_1)]$$

其中 $\hat{\boldsymbol{\theta}}_0$ 和 $\hat{\boldsymbol{\theta}}_1$ 分别为嵌套模型和完整模型的参数估计。

### 7.2 残差分析
计算标准化残差：
$$r_i = \frac{\log \hat{t}_i - \mu(X_i)}{\hat{\sigma}}$$

检验残差的分布是否符合模型假设。

## 8. 数值实现要点

### 8.1 数值稳定性
- 使用 $\log(\max(y, 10^{-10}))$ 避免对数函数的数值问题
- 限制指数函数的输入范围，避免溢出
- 使用多种优化算法确保收敛

### 8.2 参数约束
- $\sigma > 0$：尺度参数必须为正
- $|\beta_j| < 5$：回归系数的合理范围
- $-10 < \beta_0 < 10$：截距的合理范围

### 8.3 初值选择
使用数据驱动的初值：
- $\beta_0 = \log(\bar{t})$：时间均值的对数
- $\beta_1 = \beta_2 = 0$：协变量系数初值为0
- $\sigma = \max(0.5, s_t/\bar{t})$：基于时间变异系数的尺度参数

## 9. 结果解释

### 9.1 系数解释
在AFT模型中，回归系数的解释为：
- $\exp(\beta_1)$：BMI每增加1个标准差时，达标时间的倍数变化
- $\exp(\beta_2)$：年龄每增加1个标准差时，达标时间的倍数变化

### 9.2 临床意义
- **最佳时点**：在该时间进行NIPT检测，预期成功率达到目标水平
- **风险分层**：根据最佳时点对孕妇进行风险分类，指导临床决策
- **个体化建议**：基于孕妇的BMI和年龄提供个性化的检测时间建议

## 10. 模型局限性与改进方向

### 10.1 模型假设
- 假设协变量效应为对数线性
- 假设删失机制为非信息性删失
- 假设误差项服从特定分布

### 10.2 潜在改进
- 考虑非线性协变量效应（如样条函数）
- 引入更多协变量（如孕妇体重、胎儿性别等）
- 使用半参数模型减少分布假设
- 考虑竞争风险模型

---

**注：** 本文档基于 `interval_censored_survival_model.py` 代码的数学建模过程整理，详细描述了从数据预处理到模型应用的完整数学框架。