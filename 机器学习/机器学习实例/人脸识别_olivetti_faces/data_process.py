from PIL import Image
import numpy as np
from keras import backend as K

img_row, img_col = 57, 47


def load_data():
    # 图片1140*942:20*20张小图
    # 每张小图：2679 = 57*47 = （1140/20）*（942/20）
    img = Image.open('olivettifaces.gif')
    img_ndarray = np.asarray(img, dtype='float64') / 255  # 归一化
    data_faces = np.empty((400, 2679))
    for i in range(20):
        for j in range(20):
            data_faces[i * 20 + j] = np.ndarray.flatten(img_ndarray[i * 57:(i + 1) * 57, j * 47:(j + 1) * 47])

    # 设置标签
    label = np.empty((400), dtype='int')  # (400,)
    for i in range(400):
        label[i * 10:(i + 1) * 10] = i

    # 数据集划分；每个人前8张训练，第9张验证，第10张测试
    x_train = np.empty((320, 2679))
    x_val = np.empty((40, 2679))
    x_test = np.empty((40, 2679))
    y_train = np.empty(320)
    y_val = np.empty(40)
    y_test = np.empty(40)
    for i in range(40):
        x_train[i * 8:i * 8 + 8] = data_faces[i * 10:i * 10 + 8]
        x_val[i] = data_faces[i * 10 + 8]
        x_test[i] = data_faces[i * 10 + 9]
        y_train[i * 8:i * 8 + 8] = label[i * 10:i * 10 + 8]
        y_val[i] = label[i * 10 + 8]
        y_test[i] = label[i * 10 + 9]
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    return [(x_train, y_train), (x_val, y_val), (x_test, y_test)]


def channel_change(x_train, y_train, x_val, y_val, x_test, y_test):
    if K.image_data_format() == 'channel_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_row, img_col)
        x_val = x_val.reshape(x_val.shape[0], 1, img_row, img_col)
        x_test = x_test.reshape(x_test.shape[0], 1, img_row, img_col)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_row, img_col, 1)
        x_val = x_val.reshape(x_val.shape[0], img_row, img_col, 1)
        x_test = x_test.reshape(x_test.shape[0], img_row, img_col, 1)
    return [(x_train, y_train), (x_val, y_val), (x_test, y_test)]


if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    print('训练集大小,数据：', x_train.shape, '标签：', y_train.shape)
    print('验证集大小,数据：', x_val.shape, '标签：', y_val.shape)
    print('测试集大小,数据：', x_test.shape, '标签：', y_test.shape)
    list = channel_change(x_train, y_train, x_val, y_val, x_test, y_test)
    print('改变channel后:\n训练集大小,数据：', list[0][0].shape, '标签：', list[0][1].shape)
    print('验证集大小,数据：', list[1][0].shape, '标签：', list[1][1].shape)
    print('测试集大小,数据：', list[2][0].shape, '标签：', list[2][1].shape)
