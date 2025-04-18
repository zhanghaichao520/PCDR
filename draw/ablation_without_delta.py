import matplotlib.pyplot as plt

alpha = [3, 5, 6, 7, 8.5, 9, 9.5, 10, 10.5, 10.8, 11, 13, 15, 17, 20];
hit20_alpha = [0.1339, 0.1343, 0.1336, 0.1441, 0.1489, 0.1574, 0.17, 0.176, 0.1672, 0.1176, 0.1181, 0.1234, 0.1311, 0.1387, 0.1191];
beta = [0, 0.095, 0.099, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5];
hit20_beta = [0.2995, 0.3141, 0.3098, 0.2841, 0.2609, 0.275, 0.2218, 0.2159, 0.2331, 0.2316, 0.2475, 0.2594, 0.2627, 0.2394];

interactionsRange = ['G1', 'G2', 'G3', 'G4', 'G5']
percentageData = [34.71, 27.79, 16.01, 10.22, 11.01];
HR20_wo_Lsim = [36.89, 65.23, 69.64, 81.76, 94.55];
HR20_wo_Lcausal = [78.67, 66.36, 83.94, 96.36, 95.45];
HR20 = [87.58, 89.09, 88.18, 95.89, 99.09];

plt.figure(figsize=(10,2))
plt.subplot(131)
# plt.subplot(611)
# plt.get_cmap('viridis')
plt.plot(alpha, hit20_alpha, label='Hit@20', marker='o',ms=4, lw=1.0,ls="-",  color=[111/255,53/255,158/255])
# plt.yticks(range(0, 110, 20))
# plt.xticks(range(1, 5, 1))
plt.tick_params(axis='x', width=0)
plt.title("Causal Loss Param("+ chr(945)+")")
plt.legend(prop = {'size':7})

plt.subplot(132)
# plt.subplot(612)
# plt.get_cmap('viridis')
plt.plot(beta, hit20_beta, label='Hit@20', marker='o',ms=4, lw=1.0,ls="-", color=[242/255,177/255,135/255])
# plt.yticks(range(0, 110, 20))
# plt.xticks(range(1, 5, 1))
plt.tick_params(axis='x', width=0)
plt.title("Similar Loss Param("+ chr(946)+")")
plt.legend(prop = {'size':7})

plt.subplot(133)
# plt.get_cmap('viridis')
plt.plot(interactionsRange, HR20_wo_Lsim, label='HR@20 '+chr(946)+'=0', marker='o',ms=4, lw=1.0,ls="-", color=[242/255,177/255,135/255])
plt.plot(interactionsRange, HR20_wo_Lcausal, label='HR@20 '+chr(945)+'=0', marker='o',ms=4, lw=1.0,ls="--", color=[111/255,53/255,158/255])
plt.plot(interactionsRange, HR20, label='HR@20 ', marker='o',ms=4, lw=1.0,ls=":", color=[255/255,0/255,0/255])

# 创建次坐标轴用于柱状图
ax1 = plt.gca()  # 获取当前的Axes对象
ax2 = ax1.twinx()  # 创建共享x轴的次坐标轴

# 绘制柱状图
bars = ax2.bar(interactionsRange, percentageData, color='gray', alpha=0.3, width=0.4)  # alpha设置透明度，width设置柱宽
# 在每个柱子上方添加数值
idx = 0
for bar in bars:
    height = bar.get_height()
    if idx == 0:
        idx = 1
        height -= 2
        ax2.text(bar.get_x() + bar.get_width() / 2. + 0.1, height, f'{height:.2f}%', ha='center', va='bottom', fontsize=7)
        continue
    height -= 1
    ax2.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}%', ha='center', va='bottom', fontsize=7)

# ax2.set_ylabel('group percentage(%)', color='black')  # 设置右侧y轴标签
ax1.legend(prop = {'size':6.5}, ncol=2)  # 显示图例

plt.tick_params(axis='x', width=0)
plt.title('Impact of Loss Function')
plt.legend(prop = {'size':7})

plt.subplots_adjust(left=0.07, bottom=None, right=0.97, top=None,
                wspace=0.25, hspace=0.35)
# plt.savefig('/Users/hebert/Desktop/ablation.svg', format='svg')
plt.show()