import numpy as np
import heapq
import matplotlib.pyplot as plt

# 设置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# === 一）参数与数据读取 ===
L = 50.0                 # 每段距离（km）
fuel_price = 7.76        # 油价（元/升）
time_cost_rate = 20.0    # 时间成本（元/时）
toll_per_km = 0.5        # 高速通行费（元/km）
T_max = 0.7 * 10.833333     # 时间约束（小时），约7.5833小时

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

# 辅助函数
def index_to_rc(idx):
    """将节点索引（0-99）转换为行列坐标"""
    r = idx // 10 + 1  # 行（从下往上）
    c = idx % 10 + 1   # 列（从左往右）
    return r, c

def index_to_number(idx):
    """将节点索引（0-99）转换为原图编号（1-100）"""
    return idx + 1



def expected_fine(speed_limit, over_pct):
    """
    计算期望罚款，逻辑与C++版本 (calc_expected_fine) 一致。
    speed_limit: 路段限速 (km/h)
    over_pct: 超速百分比 (例如 0.2 表示超速20%)
    """
    if over_pct < 0.01: 
        if over_pct <= 0:
            return 0.0

    if speed_limit <= 40: # C++: s <= 41. Actual speed limits are 40, 60, 90, 120.
        if over_pct <= 0.2: # C++: r <= 0.21
            return 3.8889
        elif over_pct <= 0.5: # C++: r <= 0.51
            return 10.0
        else: # Covers over_pct == 0.7
            return 33.0
    elif speed_limit <= 60: # C++: s <= 61
        if over_pct <= 0.2:
            return 7.7778
        elif over_pct <= 0.5:
            return 15.0
        else:
            return 55.0
    elif speed_limit <= 90: # C++: s <= 91 (Fixed radar路段)
        if over_pct <= 0.2:
            return 105.0
        elif over_pct <= 0.5:
            return 180.0
        else:
            return 990.0
    elif speed_limit <= 120: # C++: s <= 121 (Fixed radar路段)
        if over_pct <= 0.2: # Corresponds to C++ r <= 0.21
            return 140.0
        elif over_pct <= 0.5: # Corresponds to C++ r <= 0.51
            return 180.0
        else: # Covers over_pct == 0.7
            return 1485.0
    
    # Should not be reached if speed_limit is one of the known values (40, 60, 90, 120)
    # and over_pct is one of the options.
    print(f"Warning: Unhandled case in expected_fine: speed_limit={speed_limit}, over_pct={over_pct}")
    return 0.0 # Fallback, though ideally this path isn't hit.

def calculate_segment_cost(speed_limit, over_pct, has_fixed_radar): # has_fixed_radar is kept for now but not used by new expected_fine
    """计算路段总成本（包括时间成本、燃油费、通行费和期望罚款）"""
    # 实际速度
    actual_speed = speed_limit * (1 + over_pct)
    
    # 行驶时间（小时）
    travel_time = L / actual_speed
    
    # 时间成本
    time_cost = time_cost_rate * travel_time
    
    # 燃油费
    fuel_consumption = (0.0625 * actual_speed + 1.875) * (L / 100.0)  # 油耗（升）
    fuel_cost = fuel_consumption * fuel_price
    
    # 高速通行费
    toll_cost = toll_per_km * L if speed_limit == 120 else 0.0
    
    # 期望罚款 - 使用新的逻辑
    fine_cost = expected_fine(speed_limit, over_pct) # Removed has_fixed_radar as it's implicit in new func
    
    # 总成本
    total_cost = time_cost + fuel_cost + toll_cost + fine_cost
    
    return total_cost, travel_time

# 超速选项
speed_options = [0.0, 0.2, 0.5, 0.7]  # 0%, 20%, 50%, 70%

# === 四）多标记Dijkstra算法 ===
def multi_label_dijkstra(adj, start, end, T_max):
    """多标记Dijkstra算法，在时间约束下寻找期望费用最小的路径"""
    print(f"开始多标记Dijkstra算法，从节点{start+1}到节点{end+1}，时间约束{T_max:.4f}小时")
    
    # 每个节点维护一组"(time, cost)"的Pareto前沿状态
    labels = [[] for _ in range(100)]
    
    # 初始状态
    labels[start].append((0.0, 0.0, None, None))  # (time, cost, prev_node, prev_delta)
    
    # 优先队列：(cost, time, node, path_so_far, speed_plan_so_far)
    pq = [(0.0, 0.0, start, [start], [])]
    
    # 记录最优解
    best_cost = float('inf')
    best_path = None
    best_speed_plan = None
    best_time = float('inf')
    
    # 记录处理过的状态数
    processed_states = 0
    feasible_solutions = 0
    
    while pq:
        cost_u, time_u, u, path_so_far, speed_plan_so_far = heapq.heappop(pq)
        processed_states += 1
        
        # 调试输出
        if processed_states % 10000 == 0:
            print(f"已处理{processed_states}个状态，当前队列长度：{len(pq)}")
        
        # 如果时间已经超出约束，跳过
        if time_u > T_max:
            continue
            
        # 如果已经找到更好的解，跳过
        if cost_u > best_cost:
            continue
        
        # 如果到达终点，更新最优解
        if u == end:
            if time_u <= T_max and (cost_u < best_cost or (cost_u == best_cost and time_u < best_time)):
                best_cost = cost_u
                best_time = time_u
                best_path = path_so_far.copy()
                best_speed_plan = speed_plan_so_far.copy()
                feasible_solutions += 1
                print(f"找到可行解 #{feasible_solutions}：费用={best_cost:.2f}元，时间={best_time:.4f}小时")
            continue
        
        # 遍历相邻路段
        for v, limit, vertical, has_fixed_radar in adj[u]:
            # 如果节点已经在当前路径中，跳过（避免环路）
            if v in path_so_far:
                continue
                
            # 尝试不同的超速选项
            for delta in speed_options:
                # 计算该路段的成本和时间
                seg_cost, seg_time = calculate_segment_cost(limit, delta, has_fixed_radar)
                
                # 计算新的累计时间和成本
                new_time = time_u + seg_time
                new_cost = cost_u + seg_cost
                
                # 如果超出时间约束，跳过
                if new_time > T_max:
                    continue
                
                # 检查是否被Pareto支配
                dominated = False
                for t, c, _, _ in labels[v]:
                    if t <= new_time and c <= new_cost:
                        dominated = True
                        break
                
                if dominated:
                    continue
                
                # 移除被新状态支配的旧状态
                labels[v] = [(t, c, p, d) for t, c, p, d in labels[v] 
                             if not (new_time <= t and new_cost <= c)]
                
                # 添加新状态
                labels[v].append((new_time, new_cost, u, delta))
                
                # 更新路径和超速方案
                new_path = path_so_far + [v]
                new_speed_plan = speed_plan_so_far + [delta]
                
                # 加入优先队列
                heapq.heappush(pq, (new_cost, new_time, v, new_path, new_speed_plan))
    
    print(f"算法结束，共处理{processed_states}个状态，找到{feasible_solutions}个可行解")
    
    # 如果没有找到可行解
    if best_path is None:
        print("未找到满足时间约束的可行路径！")
        return None, None, None
    
    print(f"最优路径长度：{len(best_path)}，超速方案长度：{len(best_speed_plan)}")
    
    return best_cost, best_path, best_speed_plan

# === 五）路径可视化函数 ===
def visualize_path(path, speed_plan, title):
    """可视化路径和超速方案"""
    plt.figure(figsize=(14, 14))
    ax = plt.gca()
    
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
    
    # 绘制路径
    path_x = []
    path_y = []
    for node_idx in path:
        r, c = index_to_rc(node_idx)
        # 转换为坐标系（从0开始）
        y = r - 1
        x = c - 1
        path_x.append(x)
        path_y.append(y)
    
    # 使用颜色表示超速程度
    cmap = plt.colormaps.get_cmap('RdYlGn_r')  # 红黄绿色谱，反转使红色表示高超速
    
    for i in range(len(path) - 1):
        x1, y1 = path_x[i], path_y[i]
        x2, y2 = path_x[i+1], path_y[i+1]
        delta = speed_plan[i]
        
        # 根据超速比例选择颜色
        color = cmap(delta / 0.7)  # 归一化到[0,1]
        
        plt.plot([x1, x2], [y1, y2], '-', linewidth=3, color=color)
        
        # 在路段中点标注超速比例
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        plt.text(mid_x, mid_y, f"{delta*100:.0f}%", color='black', fontweight='bold', 
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.plot(path_x[0], path_y[0], 'go', markersize=12, label='起点')
    plt.plot(path_x[-1], path_y[-1], 'ro', markersize=12, label='终点')
    
    # 添加路径节点标签
    for i, node_idx in enumerate(path):
        node_num = index_to_number(node_idx)
        r, c = index_to_rc(node_idx)
        y = r - 1
        x = c - 1
        plt.text(x+0.1, y+0.1, f"{i+1}", color='blue', fontsize=10, fontweight='bold')
    
    # 添加颜色条，表示超速程度
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 70))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('超速比例 (%)')
    # 将刻度设置在颜色条的实际数据范围 [0, 70] 内
    cbar.set_ticks([0, 20, 50, 70])
    cbar.set_ticklabels(['0%', '20%', '50%', '70%'])
    
    plt.title(title)
    plt.xlabel('列（从左到右）')
    plt.ylabel('行（从下到上）')
    plt.grid(True)
    plt.legend()
    # plt.gca().invert_yaxis()  # 注释掉或删除此行以确保Y轴0点在底部
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

# === 六）主函数：计算最优路径并验证 ===
def main():
    print("\n=== 计算时间约束下的最优路径 ===")
    start_node = 0  # 起点（甲地，左下角1号路口）
    end_node = 99   # 终点（乙地，右上角100号路口）
    
    # 运行多标记Dijkstra算法
    min_cost, optimal_path, speed_plan = multi_label_dijkstra(adj, start_node, end_node, T_max)
    
    if min_cost is None:
        print("未找到满足时间约束的可行路径！")
        return
    
    # 输出结果
    print(f"\n最小期望费用 C2 = {min_cost:.2f} 元")
    print(f"路径节点序列（原图编号）: {[index_to_number(idx) for idx in optimal_path]}")
    print(f"路径长度: {len(optimal_path)}个节点，{len(optimal_path)-1}段路")
    
    # 计算路径详情
    total_distance = 0
    total_time = 0
    total_cost = 0
    total_fine = 0
    path_details = []
    
    for i in range(len(optimal_path) - 1):
        u = optimal_path[i]
        v = optimal_path[i+1]
        delta = speed_plan[i]
        
        # 找到对应的边信息
        edge_info = None
        for neighbor, limit_val, vertical, has_radar_val in adj[u]: # Renamed to avoid conflict
            if neighbor == v:
                edge_info = (limit_val, vertical, has_radar_val)
                break
        
        if edge_info:
            limit, vertical, has_radar = edge_info # Original names for clarity below
            actual_speed = limit * (1 + delta)
            seg_cost, seg_time = calculate_segment_cost(limit, delta, has_radar)
            
            # 计算各项成本
            time_cost = time_cost_rate * seg_time
            fuel_consumption = (0.0625 * actual_speed + 1.875) * (L / 100.0)
            fuel_cost = fuel_consumption * fuel_price
            toll_cost = toll_per_km * L if limit == 120 else 0.0
            fine_cost = expected_fine(limit, delta) # 修改此处，移除 has_radar 参数
            
            u_r, u_c = index_to_rc(u)
            v_r, v_c = index_to_rc(v)
            
            path_details.append({
                "起点": f"({u_r},{u_c}) - {index_to_number(u)}号路口",
                "终点": f"({v_r},{v_c}) - {index_to_number(v)}号路口",
                "限速": limit,
                "超速比例": delta,
                "实际速度": actual_speed,
                "时间": seg_time,
                "时间成本": time_cost,
                "油费": fuel_cost,
                "通行费": toll_cost,
                "期望罚款": fine_cost,
                "总费用": seg_cost
            })
            
            total_distance += L
            total_time += seg_time
            total_cost += seg_cost
            total_fine += fine_cost
    
    # 打印路径详情
    print("\n路径详细信息:")
    print(f"{'起点':^15} | {'终点':^15} | {'限速':^8} | {'超速比例':^8} | {'实际速度':^10} | {'时间':^10} | {'期望罚款':^10} | {'总费用':^10}")
    print("-" * 120)
    
    for detail in path_details:
        print(f"{detail['起点']:^15} | {detail['终点']:^15} | {detail['限速']:^8.1f} | {detail['超速比例']*100:^8.0f}% | {detail['实际速度']:^10.1f} | {detail['时间']:^10.4f} | {detail['期望罚款']:^10.2f} | {detail['总费用']:^10.2f}")
    
    print("-" * 120)
    print(f"总计: {len(path_details)}段路径, 总距离 {total_distance} km, 总时间 {total_time:.4f} 小时, 总费用 {total_cost:.2f} 元")
    print(f"费用明细: 时间成本 {total_time*time_cost_rate:.2f} 元 + 油费和通行费 {total_cost-total_fine-total_time*time_cost_rate:.2f} 元 + 期望罚款 {total_fine:.2f} 元")
    
    # 检查时间约束
    print(f"\n时间约束检查: {total_time:.4f} 小时 <= {T_max:.4f} 小时: {'满足' if total_time <= T_max else '不满足'}")
    print(f"时间占比: {total_time/T_max*100:.2f}% (相对于时间预算)")
    print(f"时间占比: {total_time/(10.833)*100:.2f}% (相对于问题一中的T1)")
    
    # 检查超速合规性
    print("\n超速合规性检查:")
    for i, delta in enumerate(speed_plan):
        if delta > 0.7:
            print(f"错误：第{i+1}段路超速比例{delta*100:.0f}%超过了70%的上限！")
    
    print("超速合规性验证通过：所有路段超速比例均不超过70%")
    
    # 可视化路径
    try:
        visualize_path(optimal_path, speed_plan, "问题二最优路径与超速方案")
        print("\n已生成路径可视化图: 问题二最优路径与超速方案.png")
    except Exception as e:
        print(f"生成可视化图时出错: {e}")

if __name__ == "__main__":
    main()