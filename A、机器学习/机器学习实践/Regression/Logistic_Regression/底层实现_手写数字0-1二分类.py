'''
author:fangchao
date:2019/05/14

content:logistic regression 底层实现
'''

import numpy as np
import os


# 数据加载
def loadData(path='train'):
    trainFileList = os.listdir(path)
    m = len(trainFileList)
    dataArray = np.zeros((m, 1024))
    labelArray = np.zeros((m, 1))
    for i in range(m):
        returnArray = np.zeros((1, 1024))
        fileName = trainFileList[i]
        fr = open('%s/%s' % (path, fileName))
        for j in range(32):
            lineStr = fr.readline()
            for k in range(32):
                returnArray[0, 32 * j + k] = int(lineStr[k])
        dataArray[i, :] = returnArray
        fileName0 = fileName.split('.')[0]
        label = fileName0.split('_')[0]
        labelArray[i] = int(label)
    return dataArray, labelArray


# sigmoid函数
def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))


# 梯度下降,求回归系数
def gradeAscent(dataArray, labelArray, alpha=0.01, iter=100):
    dataMat = np.mat(dataArray)  # size:m*n
    labelMat = np.mat(labelArray)  # size:m*1
    m, n = dataMat.shape
    weight = np.ones((n, 1))
    for i in range(iter):
        h = sigmoid(dataMat * weight)
        error = labelMat - h  # size:m*1
        weight = weight + alpha * dataMat.transpose() * error
    return weight


# 分类函数
def classify(dataArray, labelArray, weight):
    dataMat = np.mat(dataArray)
    labelMat = np.mat(labelArray)
    h = sigmoid(dataMat * weight)  # size:m*1
    m = len(h)
    error = 0.0
    for i in range(m):
        if h[i] > 0.5:
            print(labelMat[i], 'is classifyed as :1')
            if labelMat[i] != 1:
                error += 1
                print('error')
        else:
            print(labelMat[i], 'is classifyed as :0')
            if labelMat[i] != 0:
                error += 1
                print('error')
    print('error rate: ', '%.4f' % (error / m))


# 训练测试
def digitRecognition(trainPath, testPath, alpha=0.01, iter=10):
    data, label = loadData(trainPath)
    weight = gradeAscent(data, label, alpha, iter)  # 利用训练集求得回归系数
    data_test, label_test = loadData(testPath)
    classify(data_test, label_test, weight)  # 将此回归系数用于测试集


if __name__ == '__main__':
    digitRecognition('train', 'test')

'''
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :1
error
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[0.]] is classifyed as :0
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
[[1.]] is classifyed as :1
error rate:  0.0118
'''
