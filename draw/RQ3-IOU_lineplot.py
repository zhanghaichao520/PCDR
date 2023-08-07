import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
#matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
matplotlib.rcParams.update({'font.size': 20})
# white, whitegrid
sns.set_theme(style="white",font='Times New Roman',font_scale=4)

data = pd.read_csv(r'~/Desktop/ml-1m-IOU.csv')

data = data.T
data = data.reset_index()
# 将第一行的数据变为列名
data.columns = np.array(data).tolist()[0]
data.drop([0], inplace=True)  # 删除df的第一行多余的数据

#设定画布。dpi越大图越清晰，绘图时间越久
fig=plt.figure(figsize=(15, 12), dpi=100)

# ax1=fig.add_subplot(111)                     # 设置绘图区


sns.lineplot(x="topK", y="DMCB-matching", color="#1F77B4", lw=4, marker="*", markersize=15, data=data,  label="DMCB-matching")
sns.lineplot(x="topK", y="DMCB-conformity", color="#FF7F0E", lw=4, marker="o", markersize=15, data=data,  label="DMCB-conformity")
sns.lineplot(x="topK", y="DICE",  color="#2CA02C", lw=4, marker="h", markersize=15, data=data,  label="DICE")
sns.lineplot(x="topK", y="CausE",  color="#D62728", marker="h", markersize=15,  lw=4, data=data,  label="CausE")
sns.lineplot(x="topK", y="DICE",  color="#9467BD", marker="h", markersize=15,  lw=4, data=data,  label="DICE")
sns.lineplot(x="topK", y="DCCL",  color="#8C564B", marker="h", markersize=15,  lw=4, data=data,  label="DCCL")
sns.lineplot(x="topK", y="MF_IPS",  color="#E377C2", marker="h", markersize=15,  lw=4, data=data,  label="MF_IPS")
sns.lineplot(x="topK", y="MACR",  color="#7F7F7F", marker="h", markersize=15,  lw=4, data=data,  label="MACR")

plt.ylabel("IOU (%)")
# plt.xticks([],None)
plt.legend(ncol=3,fontsize=20, loc="best")

# plt.show()
plt.savefig("/Users/hebert/Desktop/ml-1m-IOU.png",dpi=800)


