import pandas as pd
import numpy as np

# 交通法规罚款函数
def fine_amount(v_max, x):
    """
    根据限速和超速百分比计算罚款金额。
    v_max: 路段限速 (km/h)
    x: 超速百分比 (例如 0.2 表示超速20%)
    """
    if x == 0.0:  # 没有超速
        return 0.0

    if v_max <= 50:
        if x <= 0.2:  # 超出至多20%
            return 50.0
        elif x <= 0.5:  # 超过20%至50%
            return 100.0
        elif x <= 0.7:  # 超过50%至70%
            return 300.0
        else:  # 超过70% (根据题目，超速最多不超过70%，此为额外情况)
            return 500.0
    elif v_max <= 80:  # 限速50至80公里 (50 < v_max <= 80)
        if x <= 0.2:
            return 100.0
        elif x <= 0.5:
            return 150.0
        elif x <= 0.7:
            return 500.0
        else:
            return 1000.0
    elif v_max <= 100:  # 限速80至100公里 (80 < v_max <= 100)
        if x <= 0.2:
            return 150.0
        elif x <= 0.5:
            return 200.0
        elif x <= 0.7:
            return 1000.0
        else:
            return 1500.0
    else:  # 限速100公里以上 (v_max > 100)
        # 注意：法规中 "超出至多50%" 包含了 x=0.2 和 x=0.5 的情况
        if x <= 0.5:
            return 200.0
        elif x <= 0.7:  # 速度超出50%至70%
            return 1500.0
        else:  # 超过70%
            return 2000.0

# --- 1.1 读取路段数据 ---
# 假设CSV文件位于上一级目录的data文件夹中
try:
    limits_col = pd.read_csv('../data/limits_col.csv', header=None).values  # 9×10
    limits_row = pd.read_csv('../data/limits_row.csv', header=None).values  # 10×9
except FileNotFoundError:
    print("错误：找不到 ../data/limits_col.csv 或 ../data/limits_row.csv 文件。")
    print("请确保这些文件存在于 d:/Code/mm/data/ 目录下。")
    exit()


edges = []
eid = 0
# 竖向边（向上）
# 节点编号从1开始，r, c 为0-indexed的网格坐标
# (r,c) -> (r+1,c)
for r in range(9):  # 0..8
    for c in range(10): # 0..9
        # 路口编号：(行号 * 列数 + 列号 + 1)
        # 起点 u: 第 r 行, 第 c 列
        # 终点 v: 第 r+1 行, 第 c 列
        u_node = r * 10 + c + 1
        v_node = (r + 1) * 10 + c + 1
        vmax = limits_col[r, c]
        edges.append((eid, u_node, v_node, int(vmax)))
        eid += 1

# 横向边（向右）
# (r,c) -> (r,c+1)
for r in range(10): # 0..9
    for c in range(9):  # 0..8
        # 起点 u: 第 r 行, 第 c 列
        # 终点 v: 第 r 行, 第 c+1 列
        u_node = r * 10 + c + 1
        v_node = r * 10 + (c + 1) + 1
        vmax = limits_row[r, c]
        edges.append((eid, u_node, v_node, int(vmax)))
        eid += 1

# 检查
print(f'共计 {len(edges)} 条边，edge_id 范围 0~{len(edges)-1}')

# --- 1.2 定义参数与速率档 ---
BRACKETS = [0.0, 0.2, 0.5, 0.7]  # 超速档：不超速、超速20%、50%、70%

EDGE_LEN = 50        # km，每段固定长度
GAS_PRICE = 7.76     # 元/升
C_FOOD = 20          # 元/小时（餐饮/住宿/游览）
TOLL_PER_KM = 0.5    # 高速公路每公里收费

# --- 1.4 用 Pandas 构造最终表 ---
# 先把 edges 转为 DataFrame
edges_df = pd.DataFrame(edges, columns=['eid', 'u', 'v', 'vmax'])

# 雷达平均数 (rho)
# 20个移动雷达，180条边
rho = 20 / len(edges_df)

# 遍历所有 (edge, bracket)，计算 t 和 c
records = []
for _, row in edges_df.iterrows():
    eid, vmax = int(row.eid), int(row.vmax) #确保为整数类型
    has_fix = int(vmax >= 90)  # 固定雷达：如果 vmax >= 90，该路段上必有一个固定雷达

    for x in BRACKETS:
        if vmax == 0: # 如果路段限速为0 (例如，不存在的路段或错误数据)
            v_real = 0
            t_hr = float('inf') # 行驶时间无穷大
        elif (1 + x) <= 1e-9: # 避免除以零 (虽然vmax*(1+x)不太可能为零除非vmax为零)
             v_real = 0
             t_hr = float('inf')
        else:
            v_real = vmax * (1 + x)
            if v_real == 0: # 再次检查，避免除零
                t_hr = float('inf')
            else:
                t_hr = EDGE_LEN / v_real

        # 油费
        # 每百公里耗油量 V = 0.0625 * v_real + 1.875 (L/100km)
        # 对于 50 km 路段，耗油量为 V * 0.5 升
        if v_real == 0: # 如果实际速度为0，油耗也应处理
            gas_cost = 0.0 if t_hr == float('inf') else float('inf') # 或者根据具体业务逻辑定义
        else:
            V100 = 0.0625 * v_real + 1.875  # L/100km
            gas_liters = V100 * (EDGE_LEN / 100) # 50km路段的油耗
            gas_cost = gas_liters * GAS_PRICE

        # 餐饮/住宿/游览费
        food_cost = C_FOOD * t_hr if t_hr != float('inf') else float('inf')

        # 高速通行费
        # 只有当 vmax=120 时才收费
        toll_cost = EDGE_LEN * TOLL_PER_KM if vmax == 120 else 0.0

        # 雷达探测概率
        if x == 0:  # 不超速则不被探测
            P_det = 0.0
        else:
            # 单台雷达对超速 x 的检测概率 p_single
            if x <= 0.2:
                p_single = 0.7
            elif x <= 0.5:
                p_single = 0.9
            else:  # x <= 0.7
                p_single = 0.99
            
            # 期望雷达数 n_exp = (是否有固定雷达) + rho
            n_exp = has_fix + rho
            
            # “至少一个”探测到的概率 P_det = 1 - (1 - p_single)^n_exp
            P_det = 1 - (1 - p_single) ** n_exp

        # 罚款
        F = fine_amount(vmax, x)
        exp_fine = P_det * F

        # 总期望费用
        if t_hr == float('inf'):
            total_cost = float('inf')
        else:
            total_cost = gas_cost + food_cost + toll_cost + exp_fine

        records.append({
            'eid': eid,
            'bracket': x,
            't_hr': t_hr,
            'cost': total_cost,
            'gas_cost': gas_cost,
            'food_cost': food_cost,
            'toll_cost': toll_cost,
            'exp_fine': exp_fine
        })

# 构造 DataFrame
tc_df = pd.DataFrame.from_records(records)

# 设置Pandas显示选项，以便观察小数位数
pd.set_option('display.float_format', lambda val: f'{val:.4f}')

# 查看前几行
print("tc_df head:")
print(tc_df.head(8))
print(f"\n tc_df shape: {tc_df.shape}") # 应为 180*4 = 720 行

# --- 小贴士 ---
# 如果要更快地查表，可以把 tc_df 透视成一个二维数组：
# t_mat = tc_df.pivot(index='eid', columns='bracket', values='t_hr').values
# c_mat = tc_df.pivot(index='eid', columns='bracket', values='cost').values
# print("\nShape of t_mat:", t_mat.shape) # 应为 (180, 4)
# print("Shape of c_mat:", c_mat.shape) # 应为 (180, 4)

# 若要导出，直接取消注释下一行：
tc_df.to_csv('../data/edge_bracket_tc.csv', index=False, float_format='%.4f')
print("\nDataFrame tc_df (可选) 已保存到 ../data/edge_bracket_tc.csv")