'''
author:fangchao
date:2019/05/16

content:KMeans++:初始化簇时，使簇间质心尽可能远
        MiniBatchKMeans：无放回随机抽样，减少收敛时间
'''

import time
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MiniBatchKMeans, KMeans

np.random.seed(0)
batch_size = 100
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)

x_data, y_label = make_blobs(n_samples=30000, centers=centers, cluster_std=0.7)

# k-means++
k_means_pp = KMeans(init='k-means++', n_clusters=3, n_init=10)  # n_init:随机选取10次中最好（计算时长最低）的结果
k_means_pp_t0 = time.time()
k_means_pp.fit(x_data)
t0 = time.time() - k_means_pp_t0

# MiniBatchKMeans
mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size, n_init=10, max_no_improvement=10,
                      verbose=0)
mbk_t0 = time.time()
mbk.fit(x_data)
t1 = time.time() - mbk_t0

# 性能
print('k-means++ runtime:', t0)
print('MiniBatchKMeans runtime:', t1)

# 评价指标:inertia_
print('k-means++ metrics:', k_means_pp.inertia_)
print('MiniBatchKMeans metrics:', mbk.inertia_)

# 画图
import matplotlib.pyplot as plt

# KMeans++
y_hat = k_means_pp.predict(x_data)
y_centoirds = k_means_pp.cluster_centers_
plt.figure(1)
plt.text(-3, 2, 'train_time: %.2f' % t0, fontsize=10)
plt.text(-3, 1.5, 'inertia_: %.2f' % k_means_pp.inertia_, fontsize=10)
plt.scatter(x_data[:8000, 0], x_data[:8000, 1], marker='o', s=1, c=y_hat[:8000], zorder=1, alpha=0.8)
plt.scatter(y_centoirds[:, 0], y_centoirds[:, 1], marker='o', edgecolors='black', s=20, c=[0, 1, 2], zorder=2, alpha=1)
plt.xticks(())
plt.yticks(())
plt.title('KMeans')
plt.savefig('k_means++.png')
# MiniBatchKMeans
y_hat = mbk.predict(x_data)
y_centoirds = mbk.cluster_centers_
plt.figure(2)
plt.text(-3, 2, 'train_time: %.2f' % t1, fontsize=10)
plt.text(-3, 1.5, 'inertia_: %.2f' % mbk.inertia_, fontsize=10)
plt.scatter(x_data[:8000, 0], x_data[:8000, 1], marker='o', s=1, c=y_hat[:8000], zorder=1, alpha=0.8)
plt.scatter(y_centoirds[:, 0], y_centoirds[:, 1], marker='o', edgecolors='black', s=20, c=[0, 1, 2], zorder=2, alpha=1)
plt.xticks(())
plt.yticks(())
plt.title('MiniBatchKMeans')
plt.savefig('MiniBatchKMeans.png')
plt.show()

'''
k-means++ runtime: 0.28568553924560547
MiniBatchKMeans runtime: 0.1532421112060547
k-means++ metrics: 25164.97821695812
MiniBatchKMeans metrics: 25178.611517320118
'''
