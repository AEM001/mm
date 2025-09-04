
import matplotlib.pyplot as plt
import numpy as np
from set_chinese_font import set_chinese_font

# 通用字体设置
set_chinese_font()
# 示例数据
data = np.arange(10)
plt.plot(data)
plt.title('中文标题示例')
plt.xlabel('横轴：时间')
plt.ylabel('纵轴：数值')

# 显示图表
plt.show()
