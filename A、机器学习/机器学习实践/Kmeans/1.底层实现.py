'''
author:fangchao
date:2019/05/15

content:KMeans、二分KMeans的底层实现
'''

import numpy as np
import pickle
import matplotlib.pyplot as plt


class KMeans(object):
    """
    初始化质心

    n_clusters:聚类个数，即k
    initCent:质心初始化方式，可选"random"或指定一个具体的array,默认random，即随机初始化
    max_iter:最大迭代次数
    """

    def __init__(self, n_clusters=5, initCent='random', max_iter=300):
        if hasattr(initCent, '__array__'):
            n_clusters = initCent.shape[0]
            self.centroids = np.asarray(initCent, dtype=np.float)
        else:
            self.centroids = None

        self.n_clusters = n_clusters  # 簇的数量
        self.max_iter = max_iter
        self.initCent = initCent  # 初始化簇的质心
        self.clusterAssment = None  # 点的簇标记（所属得簇，离簇质心距离的平方）
        self.labels = None  # 点所属的簇
        self.sse = None  # 欧式距离的平方

    # 计算两点的欧式距离
    def _distEclud(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)

    # 随机选取k个质心,必须在数据集的边界内
    def _randCent(self, X, k):
        n = X.shape[1]  # 特征维数
        centroids = np.empty((k, n))  # k*n的矩阵，用于存储质心
        for j in range(n):  # 产生k个质心，一维一维地随机初始化
            minJ = min(X[:, j])
            rangeJ = float(max(X[:, j]) - minJ)
            centroids[:, j] = (minJ + rangeJ * np.random.rand(k, 1)).flatten()
        return centroids

    # 得到质心:self.centroids
    # 簇分配结果:self.clusterAssment
    def fit(self, X):
        # 类型检查
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]  # 样本数量
        self.clusterAssment = np.empty((m, 2))  # 第一列：所属簇的索引； 第二列：该点与所属簇质心的平方误差
        if self.initCent == 'random':
            self.centroids = self._randCent(X, self.n_clusters)

        clusterChanged = True
        for _ in range(self.max_iter):
            clusterChanged = False
            for i in range(m):
                min_Dist = np.inf
                min_index = -1
                for j in range(self.n_clusters):  # 计算某一个点到所有簇质心的距离，取最小距离及并标定其所属那个簇
                    distJI = self._distEclud(self.centroids[j, :], X[i, :])  # 点到质心的欧式距离
                    if distJI < min_Dist:
                        min_Dist = distJI
                        min_index = j
                if self.clusterAssment[i, 0] != min_index:
                    clusterChanged = True
                    self.clusterAssment[i, :] = min_index, min_Dist ** 2

            if not clusterChanged:  # 当没有点发生变化时，代表已经收敛，结束迭代
                break
            for i in range(self.n_clusters):
                ptsInClust = X[np.nonzero(self.clusterAssment[:, 0] == i)[0]]  # 取出属于第i个簇的所有点
                self.centroids[i, :] = np.mean(ptsInClust, axis=0)

        self.labels = self.clusterAssment[:, 0]
        self.sse = sum(self.clusterAssment[:, 1])

    # 计算点到各个质心的距离，预测新的数据所属的簇
    def predict(self, X):
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]  # 样本数量
        preds = np.empty((m, 2))
        for i in range(m):
            minDist = np.inf
            for j in range(self.n_clusters):
                distJI = self._distEclud(self.centroids[j, :], X[i, :])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds


class biKMeans(object):
    '''
    二分KMeans算法.用到了kMeans

    初始质心：计算所有点的均值获得

    将所有数据看成是一个簇，
    当簇的数目小于k时：
        找到某一个簇划分后sse最低的簇，进行划分
    '''

    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.centroids = None
        self.clusterAssment = None
        self.labels = None
        self.sse = None

    # 计算欧式距离
    def _distEclud(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)

    def fit(self, X):
        m = X.shape[0]
        self.clusterAssment = np.zeros((m, 2))
        centroid0 = np.mean(X, axis=0).tolist()
        centList = [centroid0]
        for j in range(m):
            self.clusterAssment[j, 1] = self._distEclud(np.asarray(centroid0), X[j, :]) ** 2

        while len(centList) < self.n_clusters:
            lowestSSE = np.inf
            for i in range(len(centList)):
                pstInCurrCluster = X[np.nonzero(self.clusterAssment[:, 0] == i)[0], :]
                clf = KMeans(n_clusters=2)
                clf.fit(pstInCurrCluster)
                centroidMat, splitClustAss = clf.centroids, clf.clusterAssment  # 划分特定簇后为两簇后，得到的不同质心，
                ssesplit = sum(splitClustAss[:, 1])  # 划分后形成的两个簇的SSE
                sseNotSplit = sum(self.clusterAssment[np.nonzero(self.clusterAssment[:, 0] != i)[0], 1])  # 其他没有划分簇的SSE
                if (ssesplit + sseNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClusterAss = splitClustAss.copy()
                    lowestSSE = ssesplit + sseNotSplit
            # 划分后，其中一个子族的索引变为原族的索引，另一个子族的索引变为len(centList),然后存入centList
            bestClusterAss[np.nonzero(bestClusterAss[:, 0] == 1)[0], 0] = len(centList)
            bestClusterAss[np.nonzero(bestClusterAss[:, 0] == 0)[0], 0] = bestCentToSplit
            centList[bestCentToSplit] = bestNewCents[0, :].tolist()
            centList.append(bestNewCents[1, :].tolist())
            self.clusterAssment[np.nonzero(self.clusterAssment[:, 0] == bestCentToSplit)[0], :] = bestClusterAss

        self.labels = self.clusterAssment[:, 0]
        self.sse = sum(self.clusterAssment[:, 1])
        self.centroids = np.asarray(centList)

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]
        preds = np.empty((m,))
        for i in range(m):
            minDist = np.inf
            for j in range(self.n_clusters):
                distJI = self._distEclud(self.centroids[j, :], X[i, :])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds


if __name__ == '__main__':
    # 加载数据 x_data:(1000,2)  y_label:(1000,)
    x_data, y_label = pickle.load(open('DATASET_data.pkl', 'rb'), encoding='iso-8859-1')

    for max_iter in range(6):
        plt.figure(max_iter + 1)
        n_clusters = 10
        initCent = x_data[50:60]  # 初始化质心
        # 模型训练
        clf = KMeans(n_clusters, initCent, max_iter)
        # clf = biKMeans(n_clusters)
        clf.fit(x_data)
        cents = clf.centroids
        labels = clf.labels
        sse = clf.sse
        print('sse:', sse)

        # 画出聚类结果
        colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']  # 10类10种颜色
        for i in range(n_clusters):
            index = np.nonzero(labels == i)[0]
            x0 = x_data[index, 0]
            x1 = x_data[index, 1]
            y_i = y_label[index]
            for j in range(len(x0)):
                plt.text(x0[j], x1[j], str(int(y_i[j])), color=colors[i], fontdict={'weight': 'bold', 'size': 8},
                         zorder=1)
            plt.scatter(cents[i, 0], cents[i, 1], marker='o', color=colors[i], linewidths=12, zorder=2)
        plt.title("SSE={:.2f}".format(sse))
        plt.axis([-30, 30, -30, 30])
        plt.savefig('%d.png' % (max_iter + 1))
    plt.show()
