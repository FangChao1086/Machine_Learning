# 机器学习实例

* [keras_手写数字识别](https://github.com/FangChao1086/machine_learning/tree/master/A、机器学习/机器学习实例/keras_手写数字识别)
* [人脸识别_olivetti_faces](#人脸识别_olivetti_faces)

<span id="人脸识别_olivetti_faces"></span>
## 人脸识别_olivetti_faces
[链接：人脸识别_olivetti_faces](https://github.com/FangChao1086/Machine_learning/tree/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E4%BE%8B/%E4%BA%BA%E8%84%B8%E8%AF%86%E5%88%AB_olivetti_faces)  
### 文件说明
* data_process.py
  * 数据：olivetti_faces包含20x20张图片，大小：(1140/20) x(942/20)=57x47x2679
  * 标签：40个人，没人10张图片
  * 数据集划分：每个人前8张训练，第9张验证，第10张测试；
* cnn.py
  * create_model，创建keras模型
  * train_model，训练模型，存模型
  ```python
  # 存模型
  model.save_weights('model_weights.h5', overwrite=True
  
  # 加载模型
  model.load_weights('model_weights.h5')
  ```
  * test_model，测试模型损失得分
* run
  * 加载数据，处理输入通道
  * 将类别转换成二进制类别矩阵(one hot)
  * 模型创建，训练，保存
  * 模型测试，预测（预测准确度：accuracy）
