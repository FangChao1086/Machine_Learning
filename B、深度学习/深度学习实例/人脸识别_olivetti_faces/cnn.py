from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils
from 人脸识别_olivetti_faces.data_process import load_data, channel_change

img_row, img_col = 57, 47
nb_filters1, nb_filters2 = 20, 40
batch_size = 40
epochs = 40


def create_model(lr=0.005, decay=1e-6, momentum=0.9):
    model = Sequential()

    if K.image_data_format() == 'channels_first':
        model.add(Conv2D(nb_filters1, kernel_size=(3, 3), input_shape=(1, img_row, img_col), activation='tanh'))
    else:
        model.add(Conv2D(nb_filters1, kernel_size=(3, 3), input_shape=(img_row, img_col, 1), activation='tanh'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(nb_filters2, kernel_size=(3, 3), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units=1000, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(units=40, activation='softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def train_model(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=1)
    # 保存模型
    model.save_weights('model_weights.h5', overwrite=True)
    return model


def test_model(model, x_test, y_test):
    model.load_weights('model_weights.h5')
    score = model.evaluate(x_test, y_test, verbose=0)
    return score


if __name__ == '__main__':
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = channel_change(x_train, y_train, x_val, y_val, x_test,
                                                                          y_test)

    # 将类别向量改变为二进制类别矩阵
    Y_train = np_utils.to_categorical(y_train, 40)  # (320,)->(320,40)
    Y_val = np_utils.to_categorical(y_val, 40)  # (40,)->(40,40)
    Y_test = np_utils.to_categorical(y_test, 40)  # (40,)->(40,40)

    model = create_model()
    train_model(model, x_train, Y_train, x_val, Y_val)
    score = test_model(model, x_test, Y_test)
    print('loss score: ', score)
