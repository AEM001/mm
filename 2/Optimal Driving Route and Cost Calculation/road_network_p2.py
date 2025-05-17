import numpy as np
import heapq
import matplotlib.pyplot as plt

# === 一）参数与数据读取 ===
L = 50.0                 # 每段距离（km）
fuel_price = 7.76        # 油价（元/升）
time_cost_rate = 20.0    # 时间成本（元/时）
toll_per_km = 0.5        # 高速通行费（元/km）
T_max = 0.7 * 10.833     # 时间约束（小时），约7.5833小时

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
    返回: 邻接表 {节点索引: [(邻居索引, 限速, 是否纵向, 是否有固定雷达), ...], ...}
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
                has_fixed_radar = speed_limit >= 90  # 限速90及以上的路段安装固定雷达
                adj[u].append((v, speed_limit, True, has_fixed_radar))
            
            # 向下连接（如果不是最下行）
            if r > 1:
                v = node_index(r-1, c)
                speed_limit = limits_col[r-2][c-1]  # 注意索引转换
                has_fixed_radar = speed_limit >= 90  # 限速90及以上的路段安装固定雷达
                adj[u].append((v, speed_limit, True, has_fixed_radar))
            
            # 向右连接（如果不是最右列）
            if c < 10:
                v = node_index(r, c+1)
                speed_limit = limits_row[r-1][c-1]  # 注意索引转换
                has_fixed_radar = speed_limit >= 90  # 限速90及以上的路段安装固定雷达
                adj[u].append((v, speed_limit, False, has_fixed_radar))
            
            # 向左连接（如果不是最左列）
            if c > 1:
                v = node_index(r, c-1)
                speed_limit = limits_row[r-1][c-2]  # 注意索引转换
                has_fixed_radar = speed_limit >= 90  # 限速90及以上的路段安装固定雷达
                adj[u].append((v, speed_limit, False, has_fixed_radar))
    
    return adj

# 构建图
adj = build_graph()

# 验证图结构
def validate_graph(adj):
    """验证图结构的正确性"""
    n_nodes = len(adj)
    assert n_nodes == 100, f"节点数应为100，实际为{n_nodes}"
    
    # 检查每个节点的邻居数量
    for u in range(100):
        r, c = u // 10 + 1, u % 10 + 1  # 转换回行列坐标
        expected_neighbors = 4
        if r == 1: expected_neighbors -= 1  # 最下行没有向下的边
        if r == 10: expected_neighbors -= 1  # 最上行没有向上的边
        if c == 1: expected_neighbors -= 1  # 最左列没有向左的边
        if c == 10: expected_neighbors -= 1  # 最右列没有向右的边
        
        actual_neighbors = len(adj[u])
        assert actual_neighbors == expected_neighbors, f"节点{u+1}（行{r}列{c}）的邻居数量应为{expected_neighbors}，实际为{actual_neighbors}"
    
    # 检查固定雷达设置
    fixed_radar_count = 0
    for u in range(100):
        for v, limit, vertical, has_fixed_radar in adj[u]:
            if has_fixed_radar:
                assert limit >= 90, f"节点{u+1}到{v+1}的限速为{limit}，小于90但标记了固定雷达"
                fixed_radar_count += 1
            else:
                assert limit < 90, f"节点{u+1}到{v+1}的限速为{limit}，大于等于90但未标记固定雷达"
    
    print(f"图结构验证通过！共有{fixed_radar_count//2}条边安装了固定雷达（每条边被计算了两次）")

validate_graph(adj)

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

print("路网图构建完成，准备计算最优路径...")
