<span id="re_"></span>
# A_C、机器学习实践
* [数据集](#数据集)
  * [load_breast_cancer](#load_breast_cancer)
* [数据处理](#数据处理)
* [模型保存与加载](#模型保存与加载)
* [Regression](#Regression)
* [SVM](#SVM)
* [Decision_Tree](#Decision_Tree)
* [RandomForest](#RandomForest)
* [GBDT](#GBDT)
* [XGBOOST](#XGBOOST)
* [KMeans](#KMeans)
* [PCA](#PCA)
* [交叉验证](#交叉验证)
* [链接：pandas](https://github.com/FangChao1086/machine_learning/blob/master/机器学习/pandas.md)
* [matplotlib](#matplotlib)
* [python日期处理datetime](#python日期处理datetime)
<span id="数据集"></span>
## [数据集](#re_)
<span id="load_breast_cancer"></span>
### load_breast_cancer
```python
from sklearn.datasets import load_breast_cancer

data_cancer = load_breast_cancer()
data = data_cancer.data[:, 0:2]  # shape:569,2
target = data_cancer.target  # shape:569,
```

<span id="数据处理"></span>
## [数据处理](#re_)
### 性别_转换成数值
```python
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
```
### 缺失值填充
#### 年龄_随机森林预测
```python
from sklearn.ensemble import RandomForestRegressor

data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
age_exist = data_for_age.loc[(data.Age.notnull())]   # 年龄不缺失的数据
age_null = data_for_age.loc[(data.Age.isnull())]
x = age_exist.values[:, 1:]
y = age_exist.values[:, 0]

rfr = RandomForestRegressor(n_estimators=1000)
rfr.fit(x, y)
age_hat = rfr.predict(age_null.values[:, 1:])
data.loc[(data.Age.isnull()), 'Age'] = age_hat
```
### 数据划分_shuffle
```python
import random

# data, label; len:42000
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
(x_train, x_val) = (data[0:30000], data[30000:])
(y_train, y_val) = (label[0:30000], label[30000:])
```
### 类别标签转换_one_hot
```python
from keras.datasets import mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # y_train:(6000,)
y_train = np_utils.to_categorical(y_train, 10)  # y_train:(6000,10)
```

<span id="模型保存与加载"></span>
## [模型保存与加载](#re_)
```python
from sklearn.externals import joblib

# 模型保存
# gbr.fit(x_train, y_train)
joblib.dump(gbr, 'train_model.m')   # 保存模型

# 模型加载
gbr = joblib.load('train_model.m')
# prediction_text = gbr.prediction(x_test)
```

<span id="Regression"></span>
## [Regression](#re_)  
[链接：相关详细代码](https://github.com/FangChao1086/Machine_learning/tree/master/A、机器学习/机器学习实践/Regression)  
[链接：Logistic Regression底层实现_手写数字0-1二分类](https://github.com/FangChao1086/Machine_learning/blob/master/A、机器学习/机器学习实践/Regression/Logistic_Regression/底层实现_手写数字0-1二分类.py)  
### 回归
<span id="LinearRegression"></span>
#### LinearRegression
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 训练
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
# 参数
print(lin_reg.coef_)
print(lin_reg.intercept_)

# 预测评估
y_hat = lin_reg.predict(x_test)
mse_loss = mean_squared_error(y_test, y_hat)
print('mse:', mse_loss)
```
#### Lasso/Ridge--GridSearchCV
```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score

# 训练
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
model = Lasso()     # L1
# model = Ridge()   # L2
alpha_scan = np.logspace(-6, 2, 10)
lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_scan}, cv=5)
lasso_model.fit(x_train, y_train)
# 参数
print(lasso_model.best_params_)

# 预测评估
y_hat = lasso_model.predict(x_test)
r2 = r2_score(y_test, y_hat)
print("r2_score:", r2)
```
### 分类
#### LogisticRegression--Pipeline
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve

# 训练
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
model = Pipeline([('sc', StandardScaler()),
                  ('clf', LogisticRegression())])
model.fit(x_train, y_train)
# 参数
print(model.get_params())

# 预测评估
y_hat = model.predict(x_test)
fpr, tpr, threshold = roc_curve(y_test, y_hat)
  ## 画图
## 中文显示
# import matplotlib as mpl
# mpl.rcParams['font.sans-serif'] = [u'simHei']
# mpl.rcParams['axes.unicode_minus'] = False
```

<span id="SVM"></span>
## [SVM](#re_)
[链接：相关详细代码](https://github.com/FangChao1086/machine_learning/blob/master/A、机器学习/机器学习实践/SVM)  

<span id="Decision_Tree"></span>
## [Decision_Tree](#re_)
### 分类
#### DecisionTreeClassifier
[链接：决策树代码（包含树的构造）](https://github.com/FangChao1086/machine_learning/blob/master/A、机器学习/机器学习实践/RandomForest/decision_tree/decision_tree.py)  
1    | low       | sunny     | yes       | yes       
2    | high      | sunny     | yes       | no        
3    | med       | cloudy    | yes       | no        
4    | low       | raining   | yes       | no        
5    | low       | cloudy    | no        | yes       
6    | high      | sunny     | no        | no        
7    | high      | raining   | no        | no        
8    | med       | cloudy    | yes       | no        
9    | low       | raining   | yes       | no        
10   | low       | raining   | no        | yes       
11   | med       | sunny     | no        | yes       
12   | high      | sunny     | yes       | no   
 ![树的构造](https://i.ibb.co/RvNhfWy/image.png)

<span id="RandomForest"></span>
## [RandomForest](#re_)  
[链接：相关详细代码](https://github.com/FangChao1086/Machine_learning/tree/master/A、机器学习/机器学习实践/RandomForest)  
### 分类
#### RandomForestClassifier  
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
clf = RandomForestClassifier(n_estimators=500, max_depth=6,
                             min_samples_leaf=3, criterion="gini",
                             random_state=1, n_jobs=-1)
clf.fit(X_train, y_train)

# 预测评估
prediction = clf.predict(X_test)
acc = accuracy_score(y_test, prediction)
print('The accuracy of Random Forest is {}'.format(acc))
```

<span id="GBDT"></span>
## [GBDT](#re_)
[链接：相关详细代码](https://github.com/FangChao1086/Machine_learning/blob/master/A、机器学习/机器学习实践/GBDT)  
### 回归
#### GradientBoostingRegressor
```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# 训练
x_train, x_test, y_train, y_test = train_test_split(load_boston().data, load_boston().target,
                                                    test_size=0.2, random_state=1)
reg_model = GradientBoostingRegressor(
    loss='ls',
    learning_rate=0.01,
    n_estimators=180,
    subsample=0.8,
    max_features=0.8,
    max_depth=4,
    verbose=2
)
reg_model.fit(x_train, y_train)

# 预测与评估
prediction_train = reg_model.predict(x_train)
mse_train = mean_squared_error(y_train, prediction_train)
prediction_test = reg_model.predict(x_test)
mse_test = mean_squared_error(y_test, prediction_test)
print("mse_train:%f  mse_test:%f " % (mse_train, mse_test))
```
#### GradientBoostingRegressor--GridSearchCV
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# 未使用GridSearchCV
# 训练
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2)
clf = GradientBoostingRegressor(loss='quantile')
clf.fit(x_train, y_train)
# 预测评估
prediction_test = clf.predict(x_test)
mse_test = mean_squared_error(prediction_test, y_test)
print('原始_mse_test = %f' % mse_test)


# GridSearchCV
# 对boosting框架参数：对损失函数'quantile'的分位数α和权重缩放率learning_rate进行调参
param_test1 = {'alpha': np.linspace(0.3, 0.9, 7),
               'learning_rate': np.linspace(0.2, 0.9, 8)}
gr_model = GridSearchCV(
    estimator=GradientBoostingRegressor(
        n_estimators=100,
        loss='quantile', random_state=10
    ),
    param_grid=param_test1,
    iid=False,
    cv=5
)
gr_model.fit(x_train, y_train)
print('得到最佳参数：%s    最佳得分：%f' % (gr_model.best_params_, gr_model.best_score_))

# 训练，使用最佳参数
clf = GradientBoostingRegressor(loss='quantile', alpha=0.5, learning_rate=0.2)
clf.fit(x_train, y_train)

# 预测评估（GridSearchCV）
prediction_test = clf.predict(x_test)
mse_test = mean_squared_error(prediction_test, y_test)
print('GridSearchCV_mse_test = %f' % mse_test)
```
### 分类
#### GradientBoostingClassifier
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

# 训练
x_train, x_test, y_train, y_test = train_test_split(load_iris().data, load_iris().target,
                                                    test_size=0.2, random_state=1, stratify=load_iris().target)
cl_model = GradientBoostingClassifier(
    loss='deviance',
    learning_rate=0.001,
    n_estimators=50,
    subsample=0.8,
    max_features=0.8,
    max_depth=3,
    verbose=2
)
cl_model.fit(x_train, y_train)

# 预测与评估（混淆矩阵）
prediction_train = cl_model.predict(x_train)
cm_train = confusion_matrix(y_train, prediction_train)
prediction_test = cl_model.predict(x_test)
cm_test = confusion_matrix(y_test, prediction_test)
print("Confusion_matrix\n train:\n%s\ntest:\n%s" % (cm_train, cm_test))
```
### GBDT(梯度提升树简易教程)
[参考链接：GBDT(梯度提升树简易教程)](https://github.com/Freemanzxp/GBDT_Simple_Tutorial)  

 回归  
 二分类   
 多分类   
 可视化   
#### 结果：
![GBDT-regression](https://i.ibb.co/3SLfzR7/GBDT-regression.png)  

<span id="XGBOOST"></span>  
## [XGBOOST](#re_)
[链接：相关详细代码](https://github.com/FangChao1086/Machine_learning/tree/master/A、机器学习/机器学习实践/XGBOOST)
```python
import xgboost as xgb

# 训练测试集读取
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.5)
data_train = xgb.DMatrix(x_train, label=y_train)
data_test = xgb.DMatrix(x_test, label=y_test)

# 训练
watch_list = [(data_test, 'eval'), (data_train, 'train')]
param = {'max_depth': 3, 'eta': 1, 'silent': 0, 'objective': 'multi:softmax', 'num_class': 3}
bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)

# 预测
y_hat = bst.predict(data_test)
```

<span id="KMeans"></span>
## [KMeans](#re_)
[KMeans知识点](https://github.com/FangChao1086/Machine_learning/blob/master/A、机器学习/A_B、机器学习算法.md#聚类)  
[KMeans代码](https://github.com/FangChao1086/Machine_learning/tree/master/A、机器学习/机器学习实践/Kmeans)  

<span id="PCA"></span>
## [PCA](#re_)  
[PCA知识点](https://github.com/FangChao1086/Machine_learning/blob/master/通用知识点.md#降维)  

```python
'''
author:fangchao
date:2019/05/14
content:PCA
'''

import numpy as np


# 零均值化（列特征）
def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    return newData, meanVal


# 选择主成分个数（依据方差百分比）
def percentage2n(eigvals, percentage):
    sortArray = np.sort(eigvals)  # 升序
    sortArray = sortArray[-1::-1]  # 逆转，降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num


def PCA(dataMat, percentage=0.99):
    newData, meanVal = zeroMean(dataMat)
    convMat = np.cov(newData, rowvar=False)  # 协方差矩阵，rowvar为False:每一行代表一个样本；为True（默认）:每一列代表一个样本
    eigVals, eigVects = np.linalg.eig(np.mat(convMat))  # 求解特征值值与特征向量，每列代表一个特征向量
    eigValIndice = np.argsort(eigVals)  # 特征值从小到大排列，返回的是下标
    n = percentage2n(eigVals, percentage)  # 依据方差百分比，选择特征值个数
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 返回前n大的特征值的下标
    n_eigVects = eigVects[:, n_eigValIndice]  # 前n大特征值对应的特征向量
    lowDataMat = newData * n_eigVects  # 新的特征空间的数据
    reConMAt = (lowDataMat * n_eigVects.T) + meanVal  # 重构的返回的近似原始数据
    return lowDataMat, reConMAt


if __name__ == '__main__':
    a = np.linspace(1, 20, 20).reshape(4, 5)
    print('原始数据:\n', a)
    b, c = PCA(a)
    print('降维后的低维数据lowDataMat:\n', b, '\n重构后返回的近似原始数据reConMAt:\n', c)

'''
原始数据:
 [[ 1.  2.  3.  4.  5.]
 [ 6.  7.  8.  9. 10.]
 [11. 12. 13. 14. 15.]
 [16. 17. 18. 19. 20.]]
降维后的低维数据lowDataMat:
 [[-16.77050983]
 [ -5.59016994]
 [  5.59016994]
 [ 16.77050983]] 
重构后返回的近似原始数据reConMAt:
 [[ 1.  2.  3.  4.  5.]
 [ 6.  7.  8.  9. 10.]
 [11. 12. 13. 14. 15.]
 [16. 17. 18. 19. 20.]]
'''
```

<span id="交叉验证"></span>
## [交叉验证](#re_)
[交叉验证知识点](https://github.com/FangChao1086/Machine_learning/blob/master/A、机器学习/A_B、机器学习算法.md#交叉验证)    
[详细代码](https://github.com/FangChao1086/machine_learning/blob/master/A、机器学习/机器学习实践/交叉验证)  

<span id="matplotlib"></span>
## [matplotlib](#re_)
[链接：速查表_matplotlib](https://github.com/FangChao1086/machine_learning/blob/master/A、机器学习/速查表/matplotlib/matplotlib.pdf)  
[参考链接：matplotlib的线条及颜色控制](https://www.cnblogs.com/darkknightzh/p/6117528.html)  
### 中文显示
```python
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
```
### 柱状图
```python
'''
author:fangchao
date:2019/5/29

content:柱状图
'''
import numpy as np
import matplotlib.pyplot as plt

# index = np.arange(5)
index = np.linspace(0, 4, 5)
y = [20, 10, 30, 30, 40]

plt.bar(index, height=y, width=0.5, color='r')
plt.title("bar demo1")
plt.savefig('bar_demo1.png')
plt.show()
```

### 网格数据
```python
import matplotlib as mpl
import matplotlib.pyplot as plt

# 画图设置
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
# 画图
N, M = 500, 500  # 横纵各采样多少个值
x1_min, x1_max = data[:, 0].min(), data[:, 0].max()  # 第0列的范围
x2_min, x2_max = data[:, 1].min(), data[:, 1].max()  # 第1列的范围
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
# 生成x_test数据点
x_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
y_hat = model.predict(x_test)  # 预测值；模型为实际模型，可变
y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同

plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示(生成的数据)
plt.scatter(data[:, 0], data[:, 1], c=target, s=30, cmap=cm_dark)  # 实际样本标签，target是实际标签
plt.xlabel(u'X[0]', fontsize=14)
plt.ylabel(u'X[1]', fontsize=14)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid()  # 显示网络
plt.title(u'model', fontsize=17)
plt.savefig('model.png')
plt.show()
```

<span id="python日期处理datetime"></span>
## [python日期处理datetime](#re_)
```python
# 时间差
import datetime as dt

date_received = '20120301'
d = '20120305'
this_gap = (
        dt.datetime(int(d[0:4]), int(d[4:6]), int(d[6:8])) -
        dt.datetime(int(date_received[0:4]), int(date_received[4:6]), int(date_received[6:8]))).days
print('时间差this_gap:', this_gap)
```
