import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.font_manager import FontProperties

# 设置Matplotlib支持中文显示
try:
    font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
    available_font = None
    for font_name in font_names:
        try:
            # 尝试直接使用字体名称，如果系统能找到
            font_prop = FontProperties(family=font_name)
            plt.text(0.5, 0.5, "测试", fontproperties=font_prop) # 尝试渲染
            available_font = font_name
            print(f"使用字体: {font_name}")
            break
        except RuntimeError: # 通常是找不到字体时
            try: # 尝试从Windows字体目录加载
                font_prop = FontProperties(fname=f"C:\\Windows\\Fonts\\{font_name}.ttf")
                plt.text(0.5, 0.5, "测试", fontproperties=font_prop) # 尝试渲染
                available_font = font_prop.get_name() # 获取加载后的字体名
                print(f"使用字体 (从文件加载): {font_name} as {available_font}")
                break
            except:
                continue
    
    if available_font is None:
        print("警告: 未找到指定的中文字体，将使用系统默认字体。")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
        font_properties = FontProperties() # 使用默认
    else:
        plt.rcParams['font.sans-serif'] = [available_font]
        font_properties = FontProperties(family=available_font)

    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"字体设置错误: {e}。将使用系统默认字体。")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    font_properties = FontProperties()


# === 一）参数与数据读取 ===
L = 50.0                 # 每段距离（km）

# 从 CSV 文件读入限速矩阵 (与原脚本一致)
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

limits_col = np.array([list(map(float, row.split(','))) for row in limits_col_data.strip().split('\n')])
limits_row = np.array([list(map(float, row.split(','))) for row in limits_row_data.strip().split('\n')])

# 辅助函数 (与原脚本一致)
def index_to_rc(idx):
    r = idx // 10 + 1
    c = idx % 10 + 1
    return r, c

def index_to_number(idx):
    return idx + 1

def number_to_index(num):
    return num - 1

# === 二）路径可视化函数 (修改版) ===
def visualize_comparison_paths(paths_data, title_text):
    """可视化多条路径进行比较"""
    try:
        plt.figure(figsize=(16, 14)) # 稍微增大图像以便容纳图例
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
                plt.text(j, i, str(node_num), ha='center', va='center', fontsize=8, fontproperties=font_properties)
        
        # 绘制各条路径
        for path_info in paths_data:
            path_indices = path_info['indices']
            path_label = path_info['label']
            path_color = path_info['color']
            line_style = path_info.get('linestyle', '-') # 默认为实线

            path_x = []
            path_y = []
            for node_idx in path_indices:
                r, c = index_to_rc(node_idx)
                y = r - 1
                x = c - 1
                path_x.append(x)
                path_y.append(y)
            
            plt.plot(path_x, path_y, line_style, linewidth=3, color=path_color, label=path_label) # 修正：linestyle -> line_style
            
            # 标记起点和终点
            plt.plot(path_x[0], path_y[0], 'o', markersize=10, markerfacecolor=path_color, markeredgecolor='black', label=f"{path_label} 起点")
            plt.plot(path_x[-1], path_y[-1], 's', markersize=10, markerfacecolor=path_color, markeredgecolor='black', label=f"{path_label} 终点")

        plt.title(title_text, fontproperties=font_properties, fontsize=16)
        plt.xlabel('列（从左到右）', fontproperties=font_properties, fontsize=12)
        plt.ylabel('行（从下到上）', fontproperties=font_properties, fontsize=12)
        plt.grid(True)
        
        # 创建图例
        legend_prop = {'family': font_properties.get_name() if hasattr(font_properties, 'get_name') else 'SimHei', 'size': 12}
        plt.legend(prop=legend_prop, loc='upper left')
        
        # 保存图像
        output_dir = os.path.dirname(os.path.abspath(__file__))
        safe_filename = "comparison_paths.png"
        output_path = os.path.join(output_dir, safe_filename)
        
        plt.savefig(output_path, dpi=300)
        print(f"图像已保存到: {output_path}")
        plt.close()
        return True
    except Exception as e:
        print(f"绘图错误: {e}")
        return False

# === 三）主函数：绘制比较路径 ===
def main():
    try:
        # 费用最优路径数据
        path_numbers_cost = [1, 2, 12, 13, 14, 24, 25, 26, 27, 37, 38, 48, 58, 59, 69, 79, 80, 90, 100]
        min_cost = 687.0
        path_indices_cost = [number_to_index(num) for num in path_numbers_cost]

        # 时间最优路径数据
        path_numbers_time = [1, 2, 12, 13, 14, 24, 25, 35, 45, 55, 56, 57, 58, 59, 69, 79, 89, 90, 100]
        T1 = 10.83
        path_indices_time = [number_to_index(num) for num in path_numbers_time]

        paths_to_visualize = [
            {
                'indices': path_indices_cost,
                'label': f'费用最优路径 ({min_cost}元)',
                'color': 'blue',
                'linestyle': '-'
            },
            {
                'indices': path_indices_time,
                'label': f'时间最优路径 ({T1}小时)',
                'color': 'red',
                'linestyle': '--'
            }
        ]
        
        title = "费用最优路径 vs 时间最优路径对比"
        success = visualize_comparison_paths(paths_to_visualize, title)
        
        if success:
            print("路径比较可视化完成！")
            print(f"费用最优路径: {path_numbers_cost}")
            print(f"时间最优路径: {path_numbers_time}")
        else:
            print("路径比较可视化失败！")
            
    except Exception as e:
        print(f"主函数错误: {e}")

if __name__ == "__main__":
    main()