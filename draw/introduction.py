import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = pd.read_csv('~/Desktop/intro_data.csv')

# 提取数据
models = data['model']
hr_popular = data['HR@20 popular']
hr_unpopular = data['HR@20 unpopular']

# 设置画布，宽度略大于高度
fig, ax = plt.subplots(figsize=(6, 4))

# 绘制柱状图，使用更加学术的颜色：深蓝和深绿
bar_width = 0.3
index = np.arange(len(models))

bars1 = ax.bar(index, hr_popular, bar_width, label='popular items', edgecolor='black', color='#1f77b4')
bars2 = ax.bar(index + bar_width, hr_unpopular, bar_width, label='unpopular items', edgecolor='black', color='#2ca02c')

# 添加大字体标签和标题
ax.set_ylabel('HR@20', fontsize=16)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(models, fontsize=14)

# 设置边框和刻度
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

plt.yticks(fontsize=14)
legend = plt.legend(frameon=True, edgecolor='black', fontsize=12)
frame = legend.get_frame()
frame.set_edgecolor('black')
frame.set_linewidth(1)

# 调整布局和显示
fig.tight_layout()
plt.savefig("/Users/hebert/Desktop/PCDR/introduction_data.png",dpi=800)
