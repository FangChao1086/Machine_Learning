## B_B、深度学习实践

* [参考链接：各种数据集下载](https://mp.weixin.qq.com/s/mq2aCU91zcTe-lkPTiAF2g)
* [tensorflow](#tensorflow)
* [优化算法](#优化算法)
* [Keras](#Keras)

## tensorflow
[链接：tensorflow基础](https://github.com/FangChao1086/machine_learning/tree/master/深度学习/tensorflow/Tensorflow基础.ipynb)  
* 工作原理
  * 是使用**数据流图**（描述的是有向图的数值计算过程）进行数值计算的。
  * 在有向图中，节点表示数学运算，边表示传输多维数据，节点也可以被分配到计算设备上从而进行并行的执行操作
* tf.Interativesession():默认自己就是用户要操作的会话
* tf.Session():没有上面的默认，所以eval()启动计算机室需要志明使用的是哪个会话

### 线性模型与逻辑回归
[参考链接：线性模型与逻辑回归](https://blog.csdn.net/weixin_43824059/article/details/86530652)
### 多层神经网络
[链接：多层神经网络](https://github.com/FangChao1086/machine_learning/blob/master/深度学习/tensorflow/多层神经网络.ipynb)
### 多分类问题与深层神经网络
[链接：多分类问题与深层神经网络_计算图可视化](https://github.com/FangChao1086/machine_learning/blob/master/深度学习/tensorflow/多分类问题及深层神经网络.ipynb)

### 定义深层网络结构
```python
def hidden_layer(layer_input, output_depth, scope='hidden_layer', reuse=None):
    input_depth = layer_input.get_shape()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        # 注意这里的初始化方法是truncated_normal
        w = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1), shape=(input_depth, output_depth), name='weights')
        # 注意这里用 0.1 对偏置进行初始化
        b = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=(output_depth), name='bias')
        net = tf.matmul(layer_input, w) + b
        
        return net

def DNN(x, output_depths, scope='DNN', reuse=None):
    net = x
    for i, output_depth in enumerate(output_depths):
        net = hidden_layer(net, output_depth, scope='layer%d' % i, reuse=reuse)
        # 注意这里的激活函数
        net = tf.nn.relu(net)
    # 数字分为0, 1, ..., 9 所以这是10分类问题
    # 对应于 one_hot 的标签, 所以这里输出一个 10维 的向量
    net = hidden_layer(net, 10, scope='classification', reuse=reuse)
    
    return net
    
# 使用网络 构造一个4层的神经网络, 它的隐藏节点数分别为: 400, 200, 100, 10
input_ph = tf.placeholder(shape=(None, 784), dtype=tf.float32)
label_ph = tf.placeholder(shape=(None, 10), dtype=tf.int64)
dnn = DNN(input_ph, [400, 200, 100])
```

## 优化算法
* Momentum
  * `train_op = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss)`
  * [链接: Momentum](https://github.com/FangChao1086/machine_learning/blob/master/深度学习/tensorflow/Momentum.ipynb)  
* Adadelta
  * [链接：Adadelta](https://github.com/FangChao1086/machine_learning/blob/master/深度学习/tensorflow/Adadelta.ipynb)
* Adam
  * [链接：Adam](https://github.com/FangChao1086/machine_learning/blob/master/深度学习/tensorflow/Adam.ipynb)

## Keras
* [参考链接：Documentation](https://keras.io/)
* [参考链接：Example_github](https://github.com/fchollet/keras/tree/master/examples)
  * [链接：keras_手写数字识别](https://github.com/FangChao1086/machine_learning/tree/master/B、深度学习/深度学习实例/keras_手写数字识别)  
### 内置数据集加载
* [参考链接：内置数据集加载](https://keras.io/datasets/)  
### 网络建模
```python
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D
from keras.layers.core import Flatten, Dense, Activation
from keras.optimizers import SGD

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
```
### 模型存储、加载、查看、网络可视化
* [参考链接：模型的存储与加载](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)  
```python
import keras
from keras.utils.vis_utils import plot_model

# 模型存储、加载、网络可视化
model.save('model.h5')  # 存整个模型（包括结构、权重与训练配置）
origin_model = keras.models.load_model('model.h5')  # 加载整个模型
print(origin_model.summary())  # 查看模型
plot_model(origin_model, to_file='model.png')  # 神经网络可视化
```
### 模型预测
```python
# 预测类别_非one_hot
predict_test = model.predict_classes(x_test, verbose=0)  # (40,)
```
