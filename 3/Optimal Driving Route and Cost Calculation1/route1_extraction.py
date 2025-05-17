import numpy as np
import heapq
import matplotlib.pyplot as plt

# === 一）参数与数据读取 ===
L = 50.0                 # 每段距离（km）
fuel_price = 7.76        # 油价（元/升）
time_cost_rate = 20.0    # 时间成本（元/时）
toll_per_km = 0.5        # 高速通行费（元/km）

# 超速选项
speed_options = [0.0, 0.2, 0.5, 0.7]  # 0%, 20%, 50%, 70%

# 从 CSV 文件读入限速矩阵
limits_col_data = """60,90,60,60,40,40,40,60,60,120
40,60,60,60,40,90,60,60,60,120
40,60,60,60,120,90,60,40,40,120
40,60,90,60,120,60,60,60,40,120
60,60,90,60,120,60,60,60,40,40
60,40,90,60,120,60,40,40,40,40
40,40,90,40,120,40,40,40,90,40
60,60,90,40,40,60,60,60,90,60
60,60,90,60,60,60,90,60,60,90"""

limits_row_data = """90,40,40,40,40,40,40,40,40
60,90,90,40,40,60,60,40,40
40,60,60,60,60,60,40,40,40
40,60,90,90,90,60,60,60,60
60,60,60,60,60,60,60,60,40
60,60,90,90,90,90,90,90,60
40,40,60,60,40,40,40,60,40
60,60,40,40,40,40,90,90,90
120,120,120,120,40,40,90,90,90
60,60,60,60,40,60,60,60,60"""

# 将字符串转换为numpy数组
limits_col = np.array([list(map(float, row.split(','))) for row in limits_col_data.strip().split('\n')])
limits_row = np.array([list(map(float, row.split(','))) for row in limits_row_data.strip().split('\n')])

# 验证数据维度
print(f"limits_col shape: {limits_col.shape}")  # 应为 (9, 10)
print(f"limits_row shape: {limits_row.shape}")  # 应为 (10, 9)

# === 二）构造邻接表 ===
def build_graph():
    """
    构建路网图的邻接表表示
    节点编号: 1-100，对应于内部索引 0-99
    节点 (r,c) 的编号为: (r-1)*10 + c，其中 r 从下往上（1-10），c 从左往右（1-10）
    返回: 邻接表 {节点索引: [(邻居索引, 限速), ...], ...}
    """
    adj = {u: [] for u in range(100)}
    
    # 转换节点坐标到索引的辅助函数
    def node_index(r, c):
        """将行列坐标转换为节点索引（0-99）"""
        return (r-1)*10 + (c-1)
    
    # 构建邻接表
    for r in range(1, 11):  # 行（从下往上）
        for c in range(1, 11):  # 列（从左往右）
            u = node_index(r, c)
            
            # 向上连接（如果不是最上行）
            if r < 10:
                v = node_index(r+1, c)
                speed_limit = limits_col[r-1][c-1]  # 注意索引转换
                adj[u].append((v, speed_limit))
            
            # 向下连接（如果不是最下行）
            if r > 1:
                v = node_index(r-1, c)
                speed_limit = limits_col[r-2][c-1]  # 注意索引转换
                adj[u].append((v, speed_limit))
            
            # 向右连接（如果不是最右列）
            if c < 10:
                v = node_index(r, c+1)
                speed_limit = limits_row[r-1][c-1]  # 注意索引转换
                adj[u].append((v, speed_limit))
            
            # 向左连接（如果不是最左列）
            if c > 1:
                v = node_index(r, c-1)
                speed_limit = limits_row[r-1][c-2]  # 注意索引转换
                adj[u].append((v, speed_limit))
    
    return adj

# 构建图
adj = build_graph()

# 转换节点索引到行列坐标的辅助函数
def index_to_rc(idx):
    """将节点索引（0-99）转换为行列坐标"""
    r = idx // 10 + 1  # 行（从下往上）
    c = idx % 10 + 1   # 列（从左往右）
    return r, c

# 转换节点索引到原图编号的辅助函数
def index_to_number(idx):
    """将节点索引（0-99）转换为原图编号（1-100）"""
    return idx + 1

# === 三）提取问题一的最短路径（路线一）===

# 路线一的节点序列（原图编号）
route1_numbers = [1, 2, 12, 13, 14, 24, 25, 35, 45, 55, 56, 57, 58, 59, 69, 79, 89, 90, 100]

# 转换为内部索引（0-99）
route1_indices = [num-1 for num in route1_numbers]

# 提取路线一的限速序列
route1_limits = []
for i in range(len(route1_indices) - 1):
    u = route1_indices[i]
    v = route1_indices[i+1]
    
    # 在邻接表中查找对应的限速
    limit = None
    for neighbor, speed_limit in adj[u]:
        if neighbor == v:
            limit = speed_limit
            break
    
    if limit is None:
        raise ValueError(f"节点{index_to_number(u)}和节点{index_to_number(v)}之间不存在直接连接！")
    
    route1_limits.append(limit)

# 验证路线一的连通性和限速
print("\n=== 路线一（问题一最短路径）验证 ===")
print(f"节点序列（原图编号）: {route1_numbers}")
print(f"节点序列（内部索引）: {route1_indices}")
print(f"限速序列: {route1_limits}")

# 计算路线一在不超速情况下的总时间
total_time_no_speeding = sum(L / limit for limit in route1_limits)
print(f"不超速情况下的总时间: {total_time_no_speeding:.4f} 小时")

# 可视化路线一
def visualize_route1():
    plt.figure(figsize=(12, 12))
    
    # 绘制网格
    for i in range(11):
        plt.axhline(y=i, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=i, color='gray', linestyle='-', alpha=0.3)
    
    # 绘制所有节点
    for i in range(10):
        for j in range(10):
            node_idx = i * 10 + j
            node_num = index_to_number(node_idx)
            plt.plot(j, i, 'o', markersize=8, color='lightgray')
            plt.text(j, i, str(node_num), ha='center', va='center', fontsize=8)
    
    # 绘制路线一
    path_x = []
    path_y = []
    for node_idx in route1_indices:
        r, c = index_to_rc(node_idx)
        # 转换为坐标系（从0开始）
        y = r - 1
        x = c - 1
        path_x.append(x)
        path_y.append(y)
    
    plt.plot(path_x, path_y, '-', linewidth=3, color='blue')
    
    # 标记起点和终点
    plt.plot(path_x[0], path_y[0], 'go', markersize=12, label='起点')
    plt.plot(path_x[-1], path_y[-1], 'ro', markersize=12, label='终点')
    
    # 添加路径节点标签
    for i, node_idx in enumerate(route1_indices):
        node_num = index_to_number(node_idx)
        r, c = index_to_rc(node_idx)
        y = r - 1
        x = c - 1
        plt.text(x+0.1, y+0.1, f"{i+1}", color='blue', fontsize=10, fontweight='bold')
    
    plt.title("路线一（问题一最短路径）")
    plt.xlabel('列（从左到右）')
    plt.ylabel('行（从下到上）')
    plt.grid(True)
    plt.legend()
    plt.gca().invert_yaxis()  # 反转y轴，使得原点在左下角
    plt.savefig("路线一.png")
    plt.close()

# 生成路线一可视化图
visualize_route1()
print("已生成路线一可视化图: 路线一.png")

print("\n路线一提取和验证完成，准备进行分段超速优化...")
