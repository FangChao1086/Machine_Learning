"""
#description: GBDT回归模型调参
"""
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# 产生训练数据
X, y = make_friedman1(n_samples=1200, random_state=1, noise=1.0)

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

# 使用最佳参数
clf = GradientBoostingRegressor(loss='quantile', alpha=0.5, learning_rate=0.2)
clf.fit(x_train, y_train)

# 预测评估（GridSearchCV）
prediction_test = clf.predict(x_test)
mse_test = mean_squared_error(prediction_test, y_test)
print('GridSearchCV_mse_test = %f' % mse_test)
