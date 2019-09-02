from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

x_train, x_test, y_train, y_test = train_test_split(load_boston().data, load_boston().target,
                                                    test_size=0.2, random_state=1)
reg_model = GradientBoostingRegressor(
    loss='ls',
    learning_rate=0.03,
    n_estimators=200,
    subsample=0.8,
    max_features=0.8,
    max_depth=3,
    verbose=2
)
reg_model.fit(x_train, y_train)

prediction_train = reg_model.predict(x_train)
mse_train = mean_squared_error(y_train, prediction_train)
prediction_test = reg_model.predict(x_test)
mse_test = mean_squared_error(y_test, prediction_test)
print("mse_train:%f  mse_test:%f " % (mse_train, mse_test))
