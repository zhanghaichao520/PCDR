import pandas as pd

# 读取文件
df = pd.read_csv('dataset/netflix/netflix.inter', sep='\t', names=['item_id', 'user_id', 'rating', 'timestamp'])

# 计算每个item_id的交互次数
item_counts = df['item_id'].value_counts()

# 计算交互次数的平均值
avg_interaction = item_counts.mean()

# 标记每个item_id是popular还是unpopular
item_popularity = item_counts.apply(lambda x: 'popular' if x > avg_interaction else 'unpopular').to_dict()

# 划分用户
conservatives = set()
radicals = set()

for user_id, group in df.groupby('user_id'):
    popular_count = sum(group['item_id'].map(item_popularity) == 'popular')
    unpopular_count = sum(group['item_id'].map(item_popularity) == 'unpopular')

    if popular_count > unpopular_count:
        conservatives.add(user_id)
    else:
        radicals.add(user_id)

# 写入新文件
df[df['user_id'].isin(conservatives)].to_csv('dataset/netflix-conservatives/netflix-conservatives.inter', index=False, sep='\t', header=False)
df[df['user_id'].isin(radicals)].to_csv('dataset/netflix-radicals/netflix-radicals.inter', index=False, sep='\t', header=False)
