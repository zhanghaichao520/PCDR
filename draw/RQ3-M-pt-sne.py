# coding='utf-8'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import matplotlib.patheffects as pe
from sklearn.svm import SVC

import seaborn as sns
def get_data():
    file_and_label = {"Ms":0,
                      "Mt":1}
    data = []
    label = []
    for file_name, idx in file_and_label.items():
        tmp_data = torch.load("/Users/hebert/Desktop/" + file_name)

        data.extend(tmp_data)
        for i in range(len(tmp_data)):
            label.append(idx)

    data = np.array(data)
    label = np.array(label)
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_predictions(clf, axes):
    # 在X和Y轴范围内分别线型截取200个点
    x0s = np.linspace(axes[0], axes[1] , 2000)
    x1s = np.linspace(axes[2], axes[3], 2000)

    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X)
    y_pred = y_pred.reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)


def plot_embedding(data, label, ax):
    attr = np.array(["User Popupar Embedding in Matching Network","User Unpopupar Embedding in Matching Network"])
    # plt.scatter(x=data[:, 0], y=data[:, 1], s=40)
    sns.scatterplot(x=data[:, 0], y=data[:, 1], s=100,
                    hue=attr[label.astype(np.int8)],
                    palette=["#FF7F0E", "#1F77B4"],
                    style=attr[label.astype(np.int8)])
    # for i in range(len(attr)):
    #     Position of each label.
        # xtext, ytext = np.median(data[label == i, :], axis=0)
        # txt = ax.text(xtext, ytext, str(attr[i]), fontsize=16)
        # txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])

    plt.xticks([])
    plt.yticks([])
    plt.legend(ncol=1,fontsize=28)
    # plt.show()
    plt.savefig("/Users/hebert/Desktop/Ms-Mt.png",dpi=800)

def main():
    data, label, n_samples, n_features = get_data()
    print('data.shape',data.shape) 
    print('label',label)
    print('label size',len(set(label)),'个')
    print('data有',n_samples,'个样本')
    print('每个样本',n_features,'维数据')
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    print('result.shape',result.shape)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    # svc = SVC(C=10)
    # svc.fit(result, label)
    # plot_predictions(svc, [-300, 300, -300, 300])
    plot_embedding(result, label, ax)
    plt.show()

if __name__ == '__main__':
    main()
