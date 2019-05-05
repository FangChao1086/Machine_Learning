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

# 预测与评估
prediction_train = cl_model.predict(x_train)
cm_train = confusion_matrix(y_train, prediction_train)
prediction_test = cl_model.predict(x_test)
cm_test = confusion_matrix(y_test, prediction_test)
print("Confusion_matrix\n train:\n%s\ntest:\n%s" % (cm_train, cm_test))
