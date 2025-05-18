import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.font_manager import FontProperties

# 设置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 尝试多种中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['font.family'] = 'SimHei'  # 全局设置字体

# === 一）参数与数据读取 ===
L = 50.0                 # 每段距离（km）
min_cost = 687.0         # 最低费用（元）

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

# 辅助函数
def index_to_rc(idx):
    """将节点索引（0-99）转换为行列坐标"""
    r = idx // 10 + 1  # 行（从下往上）
    c = idx % 10 + 1   # 列（从左往右）
    return r, c

def index_to_number(idx):
    """将节点索引（0-99）转换为原图编号（1-100）"""
    return idx + 1

def number_to_index(num):
    """将原图编号（1-100）转换为节点索引（0-99）"""
    return num - 1

# === 二）路径可视化函数 ===
def visualize_path(path, speed_plan, title):
    """可视化路径和超速方案"""
    try:
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
        
        # 使用找到的字体设置标签
        cbar.set_label('超速比例 (%)', fontsize=12)
        cbar.set_ticks([0, 20, 50, 70])
        cbar.set_ticklabels(['0%', '20%', '50%', '70%'])
        
        # 设置颜色条刻度标签的字体
        for label in cbar.ax.get_yticklabels():
            label.set_fontsize(10)
        
        plt.title(title, fontsize=14)
        plt.xlabel('列（从左到右）', fontsize=12)
        plt.ylabel('行（从下到上）', fontsize=12)
        plt.grid(True)
        plt.legend(prop={'size': 12})
        
        # 确保输出目录存在
        output_dir = os.path.dirname(os.path.abspath(__file__))
        # 创建安全的文件名
        safe_filename = "min_cost_path.png"
        output_path = os.path.join(output_dir, safe_filename)
        
        # 保存图像
        plt.savefig(output_path, dpi=300)
        print(f"图像已保存到: {output_path}")
        plt.close()
        return True
    except Exception as e:
        print(f"绘图错误: {e}")
        return False

# === 三）主函数：绘制指定路径 ===
def main():
    try:
        # 指定的路径（原图编号）
        path_numbers = [1, 2, 12, 13, 14, 24, 25, 26, 27, 37, 38, 48, 58, 59, 69, 79, 80, 90, 100]
        
        # 转换为节点索引（0-99）
        path_indices = [number_to_index(num) for num in path_numbers]
        
        # 创建超速方案（全部为0）
        speed_plan = [0.0] * (len(path_indices) - 1)
        
        # 可视化路径
        success = visualize_path(path_indices, speed_plan, f"最低费用路径（{min_cost}元）")
        
        if success:
            print(f"路径可视化完成！")
            print(f"路径节点序列（原图编号）: {path_numbers}")
            print(f"路径长度: {len(path_numbers)}个节点，{len(path_numbers)-1}段路")
            print(f"最低费用: {min_cost} 元")
        else:
            print("路径可视化失败！")
    except Exception as e:
        print(f"主函数错误: {e}")

if __name__ == "__main__":
    main()