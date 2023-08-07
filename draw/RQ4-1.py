import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 创建示例数据
data1 = pd.read_csv(r'~/Desktop/RQ4-1.csv')
x1 = data1["alpha"]
y1 = data1["hit@20"]

data2 = pd.read_csv(r'~/Desktop/RQ4-2.csv')
x2 = data2["beta"]
y2 = data2["hit@20"]

data3 = pd.read_csv(r'~/Desktop/RQ4-3.csv')
x3 = data3["delta"]
y3 = data3["hit@20"]


# 创建画布和子图
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

# 设置子图1
axes[0].plot(x1, y1, color='red', marker='o', markerfacecolor='none', markersize=5, linewidth=1)
axes[0].set_xlabel('alpha', fontsize=12)
axes[0].set_ylabel('Hit@20', fontsize=12)
axes[0].set_title('Causal Loss Param', fontsize=14)
axes[0].tick_params(labelsize=10)
axes[0].grid(color='gray', linestyle='--', linewidth=0.5)

# 设置子图2
axes[1].plot(x2, y2, color='red', marker='o', markerfacecolor='none', markersize=5, linewidth=1)
axes[1].set_xlabel('beta', fontsize=12)
axes[1].set_ylabel('Hit@20', fontsize=12)
axes[1].set_title('Similar Loss Param', fontsize=14)
axes[1].tick_params(labelsize=10)
axes[1].grid(color='gray', linestyle='--', linewidth=0.5)

# 设置子图3
axes[2].plot(x3, y3, color='red', marker='o', markerfacecolor='none', markersize=5, linewidth=1)
axes[2].set_xlabel('delta', fontsize=12)
axes[2].set_ylabel('Hit@20', fontsize=12)
axes[2].set_title('Domain Loss Param', fontsize=14)
axes[2].tick_params(labelsize=10)
axes[2].grid(color='gray', linestyle='--', linewidth=0.5)

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
# plt.show()

plt.savefig("/Users/hebert/Desktop/RQ4.png",dpi=800)
