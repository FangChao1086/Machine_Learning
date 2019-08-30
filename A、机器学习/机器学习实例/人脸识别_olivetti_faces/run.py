from 人脸识别_olivetti_faces.data_process import load_data, channel_change
from 人脸识别_olivetti_faces.cnn import create_model, train_model, test_model
import numpy as np
from keras.utils import np_utils

if __name__ == '__main__':
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = channel_change(x_train, y_train, x_val, y_val, x_test,
                                                                          y_test)
    # 将类别向量改变为二进制类别矩阵
    Y_train = np_utils.to_categorical(y_train, 40)  # (320,)->(320,40)
    Y_val = np_utils.to_categorical(y_val, 40)  # (40,)->(40,40)
    Y_test = np_utils.to_categorical(y_test, 40)  # (40,)->(40,40)

    model = create_model()
    # 训练并保存模型，保存模型在train_model函数里面
    train_model(model, x_train, Y_train, x_val, Y_val)
    print('测试lost score：', test_model(model, x_test, Y_test))

    # 测试，加载已经训练完的模型并预测
    model.load_weights('model_weights.h5')
    predict_test = model.predict_classes(x_test, verbose=0)  # (40,)
    test_accuracy = np.mean(np.equal(y_test, predict_test))
    for i in range(40):
        if y_test[i] != predict_test[i]:
            print('第', y_test[i], '张图片出错,被误分为：', predict_test[i])
    print('实际label:\n', y_test)
    print('预测label:\n', predict_test)
    print('last_accuracy: ', test_accuracy)
