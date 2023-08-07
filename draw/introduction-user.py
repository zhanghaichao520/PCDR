import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据集，假设数据集中有两列：'user_id:token'和'item_id:token'
data = pd.read_csv('/Users/hebert/PycharmProjects/RecBole/dataset/ml-100k/ml-100k.inter'
                   ,delimiter="\t")  # 替换为您的数据集文件路径
# 计算每个用户的交互次数
user_interaction_count = data['user_id:token'].value_counts()

# 计算流行物品的交互次数平均值
popularity_threshold = data['item_id:token'].value_counts().mean()

# 过滤出流行物品的交互次数
popular_items = data['item_id:token'].value_counts().index[data['item_id:token'].value_counts() > popularity_threshold]

# 计算每个用户和流行物品的交互次数
user_popularity_interaction = data[data['item_id:token'].isin(popular_items)].groupby('user_id:token')['item_id:token'].count()

# 计算每个用户和流行物品交互次数占用户所有交互物品的百分比
percentage_interacted_with_popular_items = (user_popularity_interaction / user_interaction_count) * 100

# 创建一个唯一标识符列
data['interaction_id'] = data['user_id:token'] + '_' + data['item_id:token']

# 获取交互次数占比
interaction_percentage = data[data['item_id:token'].isin(popular_items)].groupby('interaction_id')['user_id:token'].count() / data['user_id:token'].map(user_interaction_count)

# 将数据转换成透视表，方便绘制热力图
user_item_percentage_pivot = interaction_percentage.reset_index().pivot(index='user_id:token',
                                                                         columns='interaction_id',
                                                                         values='user_id:token').fillna(0)

# 绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(user_item_percentage_pivot, cmap='YlGnBu', annot=False, fmt=".1f", linewidths=0.5)
plt.xlabel('交互标识符')
plt.ylabel('用户ID')
plt.title('不同用户与不同流行物品交互次数占比热力图')

# 显示图形
plt.show()