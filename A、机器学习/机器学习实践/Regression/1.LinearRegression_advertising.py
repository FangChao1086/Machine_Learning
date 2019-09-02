#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    path = 'DATASET_advertising.csv'

    # pandas读入
    data = pd.read_csv(path)    # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio']]
    y = data['Sales']
    print(x)
    print(y)

    # 解决画图显示中文问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 绘制1（各个特征的总体分布）
    plt.plot(data['TV'], y, 'ro', label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y, 'mv', label='Newspaer')
    plt.legend(loc='lower right')
    plt.grid()
    # plt.show()
    # #
    # 绘制2（各个特征独立的分布）
    plt.figure(figsize=(6, 6))
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # 训练
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    lin_reg = LinearRegression()
    model = lin_reg.fit(x_train, y_train)
    print(model)
    print(lin_reg.coef_)
    print(lin_reg.intercept_)

    # 预测
    y_hat = lin_reg.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print('mse: ', mse, '\trmse: ', rmse)

    # 显示预测与原始数据的关系
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.legend(loc='upper right')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.grid()
    plt.show()
