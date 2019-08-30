'''
Author:FangChao
'''
from __future__ import absolute_import
from __future__ import print_function
from keras_手写数字识别.data import load_data
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D
from keras.layers.core import Flatten, Dense, Activation
from keras.optimizers import SGD
import random
from keras.callbacks import EarlyStopping
from keras import backend as K
import keras
from keras.utils.vis_utils import plot_model

# 图片维序类型th:[samples][channels][rows][cols]
# keras默认类型tf:[samples][rows][cols][channels]；
K.set_image_dim_ordering('th')

# 数据加载
data, label = load_data()
# label转换成keras; 0-9,10个类别
nb_class = 10
label = np_utils.to_categorical(label, nb_class)
# 数据shuffle
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
(x_train, x_val) = (data[0:30000], data[30000:])
(y_train, y_val) = (label[0:30000], label[30000:])


def create_model(lr=0.01, decay=1e-6, momentum=0.9):
    model = Sequential()

    model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=(1, 1),
                     padding='valid', activation='relu', input_shape=(1, 28, 28)))  # (None, 4, 24, 24)

    model.add(Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', padding='valid'))  # (None, 8, 22, 22)
    model.add(MaxPooling2D(pool_size=(2, 2)))  # (None, 8, 11, 11)

    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', padding='valid'))  # (None, 16, 9, 9)
    model.add(MaxPooling2D(pool_size=(2, 2)))  # (None, 16, 4, 4)

    model.add(Flatten())  # (None, 256)
    model.add(Dense(units=128))  # (None, 128)
    model.add(Activation('relu'))

    model.add(Dense(units=nb_class, activation='softmax'))  # (None, nb_class)

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)  # 优化方法
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# 模型
model = create_model()
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=1)  # early stopping 返回最佳epoch对应的model# early stopping 返回最佳epoch对应的model
model.fit(x_train, y_train, batch_size=100, validation_data=(x_val, y_val), nb_epoch=5, callbacks=[early_stopping])
model.save('model.h5')

# origin_model = keras.models.load_model('model.h5')  # 模型加载
# print(origin_model.summary())  # 查看模型
# plot_model(origin_model, to_file='model.png')  # 神经网络可视化
