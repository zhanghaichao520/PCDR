import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据集，假设数据集中有两列：'user_id:token'和'item_id:token'
data = pd.read_csv('/Users/hebert/PycharmProjects/RecBole/dataset/ml-1m/ml-1m.inter'
                   ,delimiter="\t")  # 替换为您的数据集文件路径

# 计算每个物品的交互次数
item_interaction_count = data['item_id:token'].value_counts()

# 统计不同交互次数的物品数量
item_count_per_interaction = item_interaction_count.value_counts()

# 将统计结果转换为DataFrame，方便绘制散点图
item_interaction_stats = pd.DataFrame({'item_interactions': item_count_per_interaction.index, 'item_nums': item_count_per_interaction.values})

# 绘制散点图
plt.scatter(item_interaction_stats['item_nums'], item_interaction_stats['item_interactions'], s=10, alpha=0.5)

# 添加横纵坐标标签和标题
plt.xlabel('有这么多交互次数的物品数量')
plt.ylabel('物品交互次数')
plt.xlim(0, 40)
plt.ylim(0, 3500)
# 显示图形
plt.show()