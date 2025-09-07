import matplotlib.pyplot as plt
import platform
def set_chinese_font():
    system = platform.system()
    if system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['STSong', 'Songti SC', 'STHeiti']
    elif system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'STSong']  # 其他系统尝试通用字体
    plt.rcParams['axes.unicode_minus'] = False
