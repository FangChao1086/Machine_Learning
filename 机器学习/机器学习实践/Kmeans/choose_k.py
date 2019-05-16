'''
author:fangchao
date:2019/05/16

content:簇类K的选取:评估指标calinski_harabaz_score越大聚类性能越好
'''

from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

centers = [[0, 0], [1, 1], [1.9, 2], [3, 3]]
std1 = [0.19, 0.2, 0.3, 0.4]
seed1 = 0

x_data, y_label = make_blobs(n_samples=30000, centers=centers, cluster_std=std1, random_state=seed1)

k_means = KMeans(init='k-means++', n_clusters=4, n_init=10, random_state=0)
y_pred = k_means.fit_predict(x_data)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_pred)
plt.title('calinski_harabaz_score(k=4):%.2f' % metrics.calinski_harabaz_score(x_data, y_pred))
plt.savefig('calinski_harabaz_score(k=4).png')
plt.show()
