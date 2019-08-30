## 人脸识别_olivetti_faces

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
