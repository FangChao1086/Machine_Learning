'''
author:fangchao
date:2019/5/20

content:linear_svm
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import svm

# 数据
data_cancer = load_breast_cancer()
data = data_cancer.data[:, 0:2]  # shape:569,2
target = data_cancer.target  # shape:569,

# 模型
model = svm.SVC(kernel='linear', C=10000)
model.fit(data, target)

# 画图
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# plots the points
plt.scatter(data[:, 0], data[:, 1], c=target, s=30, cmap=plt.cm.prism)

# Creates the axis bounds for the grid
axis = plt.gca()
x_limit = axis.get_xlim()
y_limit = axis.get_ylim()

# Creates a grid to evaluate model
x = np.linspace(x_limit[0], x_limit[1], 50)
y = np.linspace(y_limit[0], y_limit[1], 50)
X, Y = np.meshgrid(x, y)
xy = np.c_[X.ravel(), Y.ravel()]  # shape: 2500, 2

# Creates the decision line for the data points
# use model.predict if you are classifying more than two
decision_line = model.decision_function(xy).reshape(Y.shape)  # shape：50, 50

# Plot the decision line and the margins
axis.contour(X, Y, decision_line, colors='k', levels=[0],
             linestyles=['-'])
# Shows the support vectors that determine the desision line
axis.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
             linewidth=1, facecolors='none', edgecolors='k')

# Shows the graph
plt.savefig('model.png')
plt.show()
