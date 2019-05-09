from PIL import Image
import numpy as np
import os


# 读取mnist下的42000张图片；
# 图片是灰度图片用1
# 如果图片是彩色的用3
def load_data():
    data = np.empty((42000, 1, 28, 28), dtype="float32")
    label = np.empty((42000,), dtype='float32')
    imgs = os.listdir('mnist')  # 得到各个图片的文件名
    num = len(imgs)
    for i in range(num):
        img = Image.open('mnist/' + imgs[i])
        arr = np.asarray(img, dtype='float32')
        data[i, :, :, :] = arr
        label[i] = int(imgs[i].split('.')[0])
    # 归一化 零均值化
    data /= np.max(data)
    data -= np.mean(data)
    return data, label


if __name__ == '__main__':
    data, label = load_data()
    print('-----------')
    print(data.ndim)  # 4
    print(data.shape)  # (42000， 1， 28， 28)
