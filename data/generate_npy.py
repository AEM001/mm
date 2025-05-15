import numpy as np
import pandas as pd

# 网格参数
N = 10  # 10×10 网格，节点编号 0…99
L = 50  # 路段长度（km）

# 读取文件
h = pd.read_csv(r'D:\Code\mm\data\limits_row.csv', header=None).values   # 10×9
v = pd.read_csv(r'D:\Code\mm\data\limits_col.csv', header=None).values  # 10×9 （已转置）

# 初始化：100 节点 × 4 方向（上0、右1、下2、左3）
speed_limit  = np.zeros((N*N, 4), dtype=np.int16)
highway_mask = np.zeros((N*N, 4), dtype=bool)

# ——水平边（左右方向）——
# 行 r=0（最底行）到 r=9（最顶行），
# 列 c=0…8 对应节点 u=r*N+c → w=u+1
for r in range(N):
    for c in range(N-1):
        u = r * N + c
        w = u + 1
        lim = int(h[r, c])
        speed_limit[u, 1] = lim
        speed_limit[w, 3] = lim
        if lim == 120:
            highway_mask[u, 1] = True
            highway_mask[w, 3] = True

# ——垂直边（上下方向）——
# 原连接行 r=0…8（节点行 1→2…9→10）
# 原列 c=0…9
# 因为 v 已转置，lim = v[c, r]
for r in range(N-1):
    for c in range(N):
        u = r * N + c
        w = (r + 1) * N + c
        lim = int(v[c, r])    # 取 limits_col.csv 中第 c 行、第 r 列
        speed_limit[u, 2] = lim
        speed_limit[w, 0] = lim
        if lim == 120:
            highway_mask[u, 2] = True
            highway_mask[w, 0] = True

# 校验无向边数：应有 180 条，每条两向共 360 记录
assert (speed_limit > 0).sum() == 360

# 存盘，后续可直接 np.load 使用
np.save(r'D:\Code\mm\data\speed_limit.npy', speed_limit)
np.save(r'D:\Code\mm\data\highway_mask.npy', highway_mask)

print("转换完成：speed_limit.npy 和 highway_mask.npy 已生成")

# -----------------------------------------------------------------------------
#运动过程是：在原表格中，从左下角1，到右上角100进行移动
# 1. 横向 limits_row.csv（10×9）：
#    - 文件行 r = 0…9 对应网格底部第 r 行（从下往上数）
#    - 文件列 c = 0…8 对应该行中节点 u=r*N+c → w=u+1 这条水平路段
#    - 取 lim = h[r, c]，并写入：
#        speed_limit[u,1] = lim    # u→w 方向为“右”（index 1）
#        speed_limit[w,3] = lim    # w→u 方向为“左”（index 3）
#      同时如果 lim==120，则 highway_mask[...] = True

# 2. 纵向 limits_col.csv（10×9，已做“列转行”处理）：
#    - 文件行 c = 0…9 对应网格第 c 列（从左往右数）
#    - 文件列 r = 0…8 对应第 r→r+1 行的垂直连接（从下往上数）
#    - 取 lim = v[c, r]，并写入：
#        speed_limit[u,2] = lim    # u→w 方向为“下”（index 2）
#        speed_limit[w,0] = lim    # w→u 方向为“上”（index 0）
#      同样如果 lim==120，则 highway_mask[...] = True

# 3. 节点编号映射（0-based）：
#    - u = r * N + c
#    - w = u + 1         （水平右移）
#    - w = (r+1)*N + c   （垂直下移）

# 4. 校验：
#    - 无向边共有 10×9+9×10 = 180 条，每条对应两个方向 → speed_limit>0 的总数应为 360
# -----------------------------------------------------------------------------
