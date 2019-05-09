from __future__ import print_function
import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from keras_手写数字识别.data import load_data
import random
import numpy as np


def svc(traindata, trainlabel, testdata, testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1, kernel="rbf", cache_size=2000)
    svcClf.fit(traindata, trainlabel)

    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i] == pred_testlabel[i]]) / float(num)
    print("cnn-svm Accuracy:", accuracy)


def rf(traindata, trainlabel, testdata, testlabel):
    print("Start training Random Forest...")
    rfClf = RandomForestClassifier(n_estimators=400, criterion='gini')
    rfClf.fit(traindata, trainlabel)

    pred_testlabel = rfClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i] == pred_testlabel[i]]) / float(num)
    print("cnn-rf Accuracy:", accuracy)


if __name__ == "__main__":
    # load data
    data, label = load_data()
    # shuffle the data
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]
    (traindata, testdata) = (data[0:30000], data[30000:])
    (trainlabel, testlabel) = (label[0:30000], label[30000:])

    # 原始模型
    origin_model = keras.models.load_model('model.h5')
    pred_testlabel = origin_model.predict_classes(testdata, batch_size=1, verbose=1)
    num = len(testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i] == pred_testlabel[i]]) / float(num)
    print(" Origin_model Accuracy:", accuracy)

    # 新特征作为SVM的输入
    get_feature = keras.backend.function([origin_model.layers[0].input], [origin_model.layers[6].output])

    # 解决内存溢出问题
    feature1 = get_feature([data[0:10000]])[0]
    feature2 = get_feature([data[10000:20000]])[0]
    feature3 = get_feature([data[20000:30000]])[0]
    feature4 = get_feature([data[30000:40000]])[0]
    feature5 = get_feature([data[40000:]])[0]
    feature = np.concatenate([feature1, feature2, feature3, feature4, feature5])
    # feature = get_feature([data])[0]  # 内存够用时可以直接使用全部一起读数据

    # train svm using FC-layer feature
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature)
    svc(feature[0:30000], label[0:30000], feature[30000:], label[30000:])
