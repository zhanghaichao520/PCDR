import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus']=False
matplotlib.rcParams.update({'font.size': 40})
# white, whitegrid
sns.set_theme(style="whitegrid",font='Times New Roman',font_scale=3.5)

data = pd.read_csv(r'~/Desktop/RQ4.csv')
#设定画布。dpi越大图越清晰，绘图时间越久
fig=plt.figure(figsize=(14, 9), dpi=100)
ax1=fig.add_subplot(111)                     # 设置绘图区
ax1.tick_params(axis='x', labelsize=24)


# 正确显示中文和负号
# 画图，plt.bar()可以画柱状图
for index, item in data.iterrows():
    b = ax1.bar(item["Number of interactions Range"], item["Percentage of data"], label=item["Number of interactions Range"], color="#FFA07A")
    plt.bar_label(b)

# plt.xticks([],None)
plt.ylim(0,50)
# plt.legend(ncol=1,fontsize=24, loc="upper left")

# 设置图片名称
# 设置x轴标签名
plt.xlabel("Item Group (K)", fontsize = 40)
# 设置y轴标签名
plt.ylabel("Items/Total Items (%)", fontsize = 40)

ax2 = ax1.twinx()  # mirror the ax1
ax2.set_ylim(30,115)
sns.lineplot(x="Number of interactions Range", y="HR@20 w/o Lsim", linewidth=5,
             data=data, ax=ax2, markers=True,ls="-", marker="o", markersize=7, dashes=False, label="HR@20 w/o Lsim")
sns.lineplot(x="Number of interactions Range", y="HR@20 w/o Lcausal", linewidth=5,
             data=data, ax=ax2, markers=True,ls="-", marker="o", markersize=7, dashes=False, label="HR@20 w/o Lcausal")
sns.lineplot(x="Number of interactions Range", y="HR@20 w/o Ldomain", linewidth=5,
             data=data, ax=ax2, markers=True,ls="-", marker="o", markersize=7, dashes=False, label="HR@20 w/o Ldomain")
sns.lineplot(x="Number of interactions Range", y="HR@20", linewidth=5,
             data=data, ax=ax2, markers=True,ls="-", marker="o", markersize=7, dashes=False, label="HR@20")
plt.ylabel("HR@20 (%)", fontsize = 40)
plt.legend(ncol=2, fontsize=28, loc="upper left")


# plt.axhline(y=0, color='gray', ls="--", lw=4, label='分割线')

# 显示
plt.show()
# plt.savefig("/Users/hebert/Desktop/jester.png",dpi=800)
