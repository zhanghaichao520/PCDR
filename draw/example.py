import matplotlib.pyplot as plt

task = [1, 2, 3, 4]
PIRB_1 = [78.4, 99.6, 93.3, 98.1]
PathInt_1 = [76.5, 40.7, 49.6, 98.1]
iCaRL_1 = [57.9, 98.8, 97.2, 97.3]

PIRB_2 = [88.4, 99.9, 68.8, 97.7]
PathInt_2 = [8, 5.9, 70, 98.1]
iCaRL_2 = [94.9, 100, 80.4, 96.2]

PIRB_3 = [95.2, 99.9, 87.9, 97]
PathInt_3 = [94.1, 95, 63.2, 95.4]
iCaRL_3 = [95.6, 99, 87.5, 96.2]

PIRB_4 = [83.8, 71.1, 97.4, 100]
PathInt_4 = [14.9, 34.2, 92.7, 100]
iCaRL_4 = [87.5, 79, 96.8, 100]

PIRB_5 = [74.1, 91.8, 99.2, 97.2]
PathInt_5 = [61.3, 97.1, 96.2, 97.8]
iCaRL_5 = [88.3, 95.4, 99.4, 98.1]

PIRB_6 = [99.1, 62, 78.9, 92.1]
PathInt_6 = [29.3, 69.1, 73.7, 89.1]
iCaRL_6 = [100, 59.6, 83.7, 85.6]

plt.figure(figsize=(7,4))
plt.subplot(231)
# plt.subplot(611)
plt.get_cmap('viridis')
plt.plot(task, PIRB_1, label='PIRB', marker='o',ms=7,mec='c',lw=1.0,ls="-")
plt.plot(task, PathInt_1, label='PathInt', marker='o',ms=7,mec='c',lw=1.0,ls="--")
plt.plot(task, iCaRL_1, label='iCaRL', marker='o',ms=7,mec='c',lw=1.0,ls=":")
# plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
# plt.axhline(y=sum(PIRB_1)/len(PIRB_1), color='r', linestyle='--')
# plt.axhline(y=sum(PathInt_1)/len(PathInt_1), color='b', linestyle='--')
# plt.axhline(y=sum(iCaRL_1)/len(iCaRL_1), color='g', linestyle='--')
plt.yticks(range(0, 110, 20))
plt.xticks(range(1, 5, 1))
plt.tick_params(axis='x', width=0)
plt.title('Sequence 1')
# plt.fill_between(task, min(PIRB_1), max(PIRB_1), color='r', alpha=0.2)
# plt.fill_between(task, min(PathInt_1), max(PathInt_1), color='b', alpha=0.2)
# plt.fill_between(task, min(iCaRL_1), max(iCaRL_1), color='g', alpha=0.2)
plt.legend(prop = {'size':7})

plt.subplot(232)
# plt.subplot(612)
plt.get_cmap('viridis')
plt.plot(task, PIRB_2, label='PIRB', marker='o',ms=7,mec='c',lw=1.0,ls="-")
plt.plot(task, PathInt_2, label='PathInt', marker='o',ms=7,mec='c',lw=1.0,ls="--")
plt.plot(task, iCaRL_2, label='iCaRL', marker='o',ms=7,mec='c',lw=1.0,ls=":")
# plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
plt.yticks(range(0, 110, 20))
plt.xticks(range(1, 5, 1))
plt.tick_params(axis='x', width=0)
plt.title('Sequence 2')
plt.legend(prop = {'size':7})

plt.subplot(233)
# plt.subplot(613)
plt.get_cmap('viridis')
plt.plot(task, PIRB_3, label='PIRB', marker='o',ms=7,mec='c',lw=1.0,ls="-")
plt.plot(task, PathInt_3, label='PathInt', marker='o',ms=7,mec='c',lw=1.0,ls="--")
plt.plot(task, iCaRL_3, label='iCaRL', marker='o',ms=7,mec='c',lw=1.0,ls=":")
# plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
plt.yticks(range(50, 110, 10))
plt.xticks(range(1, 5, 1))
plt.title('Sequence 3')
plt.legend(prop = {'size':7})

plt.subplot(234)
# plt.subplot(614)
plt.get_cmap('viridis')
plt.plot(task, PIRB_4, label='PIRB', marker='o',ms=7,mec='c',lw=1.0,ls="-")
plt.plot(task, PathInt_4, label='PathInt', marker='o',ms=7,mec='c',lw=1.0,ls="--")
plt.plot(task, iCaRL_4, label='iCaRL', marker='o',ms=7,mec='c',lw=1.0,ls=":")
# plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
plt.yticks(range(0, 110, 20))
plt.xticks(range(1, 5, 1))
plt.tick_params(axis='x', width=0)
plt.title('Sequence 4')
plt.legend(prop = {'size':7})

plt.subplot(235)
# plt.subplot(615)
plt.get_cmap('viridis')
plt.plot(task, PIRB_5, label='PIRB', marker='o',ms=7,mec='c',lw=1.0,ls="-")
plt.plot(task, PathInt_5, label='PathInt', marker='o',ms=7,mec='c',lw=1.0,ls="--")
plt.plot(task, iCaRL_5, label='iCaRL', marker='o',ms=7,mec='c',lw=1.0,ls=":")
# plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
plt.yticks(range(50, 110, 10))
plt.xticks(range(1, 5, 1))
plt.tick_params(axis='x', width=0)
plt.title('Sequence 5')
plt.legend(prop = {'size':7})

plt.subplot(236)
# plt.subplot(616)
plt.get_cmap('viridis')
plt.plot(task, PIRB_6, label='PIRB', marker='o',ms=7,mec='c',lw=1.0,ls="-")
plt.plot(task, PathInt_6, label='PathInt', marker='o',ms=7,mec='c',lw=1.0,ls="--")
plt.plot(task, iCaRL_6, label='iCaRL', marker='o',ms=7,mec='c',lw=1.0,ls=":")
# plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
plt.yticks(range(0, 110, 20))
plt.xticks(range(1, 5, 1))
plt.title('Sequence 6')
plt.legend(prop = {'size':7})

plt.subplots_adjust(left=0.07, bottom=None, right=0.97, top=None,
                wspace=0.25, hspace=0.35)
# plt.savefig('/Users/zhaozeyun/Desktop/On different task sequence order')
plt.savefig('/Users/hebert/Desktop/On different task sequence order.svg', format='svg')
plt.show()