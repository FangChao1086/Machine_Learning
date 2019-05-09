from __future__ import print_function
import keras
from keras_手写数字识别.data import load_data
import matplotlib.pyplot as plt

data, label = load_data()
model = keras.models.load_model('model.h5')

get_feature = keras.backend.function([model.layers[0].input], [model.layers[8].output])
get_feature_map = keras.backend.function([model.layers[0].input], [model.layers[1].output])

# 输出最后的特征
feature_ = get_feature([data[0:10]])[0]
plt.imshow(feature_[0].reshape(1, -1), cmap='gray')
plt.savefig('layer_last.png')
plt.show()

# 输出第二层特征
feature = get_feature_map([data[0:10]])[0]  # (10, 8, 22, 22)
for i in range(8):
    show_img = feature[0][i][:][:]
    plt.subplot(2, 4, i + 1)
    plt.imshow(show_img, cmap='gray')
    plt.title('kernel_%d' % (i + 1))
    plt.axis('off')

plt.savefig('layer1.png')
plt.show()
