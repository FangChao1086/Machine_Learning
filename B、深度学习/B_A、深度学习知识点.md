<span id="re_"></span>
## B_A、深度学习知识点
* [BatchNormalization](#BatchNormalization)
* [梯度消失与梯度爆炸](#梯度消失与梯度爆炸)
* [反向传播(BP)](#反向传播(BP))
* [优化算法](#优化算法)
  * [随机梯度下降](#随机梯度下降)
    * [随机梯度下降的优化算法](#随机梯度下降的优化算法)
  * [小批量随机梯度下降](#小批量随机梯度下降) 
* [LSTM](#LSTM)
* [GRU](#GRU)
* [RNN](#RNN)
* [Attention机制](#Attention机制)
* [1x1卷积作用](#1x1卷积作用)
* [Dropout](#Dropout)

<span id="BatchNormalization"></span>
## [BatchNormalization](#re_)
* 神经网络在训练的时候，随着训练层数的加深，
* **激活函数的输入值的整体分布逐渐往激活函数的取值区间上下限靠近**，从而**导致在反向传播时低层神经网络的梯度消失**。
* BN是通过规范化手段，将越来越偏的分布拉回标准化分布，使得激活函数的输入值落在对输入比较敏感的区域，避免梯度消失

<span id="梯度消失与梯度爆炸"></span>
## [梯度消失与梯度爆炸](#re_)
* 那么为什么会出现梯度消失的现象呢？  
通常神经网络所用的**激活函数是sigmoid函数**，这个函数有个特点，就是能将负无穷到正无穷的数映射到0和1之间，并且对这个函数求导的结果是f′(x)=f(x)(1−f(x))。因此两个0到1之间的数相乘，得到的结果就会变得很小了。**神经网络的反向传播是逐层对函数偏导相乘**，因此**当神经网络层数非常深的时候，最后一层产生的偏差就因为乘了很多的小于1的数而越来越小**，最终就会变为0，从而导致**层数比较浅的权重没有更新，这就是梯度消失**。
* 那么什么是梯度爆炸呢？  
梯度爆炸就是由于**初始化权值过大，前面层会比后面层变化的更快，就会导致权值越来越大**，梯度爆炸的现象就发生了。
<span id="反向传播(BP)"></span>
## [反向传播(BP)](#re_)
[参考链接：手推反向传播](https://www.cnblogs.com/makefile/p/BP.html)  
<div style="align: center"><img src="https://i.ibb.co/hYztJBh/image.png"></div>  

**上图说明**  
* 前向传播$f(x,y)=Z$，更新$Z$节点的值
* 反向传播时，$Z$作为中间的一个节点
  * 损失函数传播到$Z$节点的梯度为：
  $$\frac{\partial \operatorname{loss}}{\partial z}=\frac{\partial L}{\partial z}$$  
  * 传播到$x$,$y$点的梯度，需要以$Z$点作为中继，使用链式法则得到，见上图

<span id="优化算法"></span>
## [优化算法](#re_)
<span id="随机梯度下降"></span>
### 随机梯度下降
每次计算只使用一个样本
* 增加了跳出当前局部最小值的潜力

<span id="随机梯度下降的优化算法"></span>
#### 随机梯度下降的优化算法
[视频和pdf介绍](https://ask.julyedu.com/question/7913)
* 动量法（Momentum）
  * 每次吸收上一部分更新的余势  
  $$v_{i}=\gamma v_{i-1}+\eta \nabla_{\theta} L(\theta)$$  
  $$\theta_{i}=\theta_{i-1}-v_{i}$$  
    * 当前速度：$v_i$；动量参数：$\gamma$；学习率：$\eta$
    * 说明：每个参数在各个方向上的移动幅度不仅取决于当前的梯度，还取决于过去各个梯度在各个方向上是否一致
* 动量法的改进算法（**Nesterov** accelerated gradient）
  * 到达底部之前及时刹车
  * 预判自己的下一步位置，并到预判位置计算梯度  
  ![NAG](https://i.ibb.co/mtjSN0b/NAG.png)
* Adagrad（自动调整学习率，适用于稀疏数据）
  * 随着模型的训练，学习率自动衰减 
  * 对于更新频繁的参数，采取较小的学习率
  * 对于更新不频繁的参数，采取较大的学习率
  * **对每个参数历史的每次更新进行叠加，并以此来做下一次更新的惩罚系数**  
  ![Adagrad](https://i.ibb.co/yXsKS00/Adagrad.png)
* Adadelta（Adagrad的改进算法）
  * 使用梯度平方的**移动平均**来替代全部历史平方和  
  ![Adadelta1](https://i.ibb.co/HF28PgZ/Adadelta1.png)
  * 为了解决梯度与参数的单位不匹配的问题，使用参数更新的移动平均来取代学习率  
  ![Adadelta2](https://i.ibb.co/rvcDkZ6/Adadelta2.png)
* Adam（RMSProp + Momentum）
  * 一阶矩（梯度自身的求和）：动量
  * 二阶矩（梯度的平方和）：自动衰减学习率  
  ![Adam](https://i.ibb.co/6sNT1w5/Adam.png)
  
<div align=center><img src="https://github.com/FangChao1086/Machine_Learning/blob/master/依赖文件/SGD优化1.png"/></div>  
<div align=center><img src="https://github.com/FangChao1086/Machine_Learning/blob/master/依赖文件/SGD优化2.png"/></div>  

<span id="小批量随机梯度下降"></span>
### 小批量随机梯度下降
每次梯度计算使用一个小批量样本

<span id="LSTM"></span>
## [LSTM->一种特殊的 RNN 类型](#re_)
* 设计初衷：避免长期依赖问题，记住长期的信息
* **输入门、遗忘门和输出门来控制输入值、记忆值和输出值**  

![LSTM](https://i.ibb.co/vs4x5sn/LSTM.png)  
### LSTM分解步骤
* 遗忘门：$f_t$；                     **决定丢弃信息**  
![LSTM_1](https://i.ibb.co/GC8Dhjt/LSTM-1.png)  
* 输入门：$i_t$，候选记忆单元：$\tilde{C_t}$；  **确定更新信息**  
![LSTM_2](https://i.ibb.co/gTkR7tY/LSTM-2.png)  
* 当前时刻记忆单元$C_t$；            **更新细胞状态**  
![LSTM_3](https://i.ibb.co/GtYBZKD/LSTM-3.png)  
* 输出门：$o_t$，输出：$h_t$  
![LSTM_4](https://i.ibb.co/72XpDfy/LSTM-4.png)  

<span id="GRU"></span>
## [GRU->LSTM的变种](#re_)
* 将LSTM网络中的**遗忘门和输入门**用**更新门**来替代
* 也是可以解决RNN网络中的长时依赖问题
* **更新门和重置门**  
![GRU](https://i.ibb.co/4gjb1rc/GRU.png)  
* **zt:更新门**
  * 用于控制前一时刻的状态信息被带入到当前状态中的程度，更新门的值越大说明前一时刻的状态信息带入越多。
* **rt:重置门**
  * 控制前一状态有多少信息被写入到当前的候选集 ~h_t上，重置门越小，前一状态的信息被写入的越少
* ~h_t:候选记忆单元
* h_t:当前时刻记忆单元

**LSTM和GRU解决RNN梯度消失问题的原因**
* RNN的某一单元主要受它附近单元的影响
* LSTM通过阈门记忆一些长期的信息
* GRU通过重置和更新两个阈门保留长期记忆

<span id="Attention机制"></span>
## [Attention机制](#re_)
* 作用：
  * 减少处理高维输入数据的计算负担,结构化的选取输入的子集,从而降低数据的维度。让系统更加容易的找到输入的数据中与当前输出信息相关的有用信息,从而提高输出的质量。帮助类似于decoder这样的模型框架更好的学到多种内容模态之间的相互关系。

<span id="RNN"></span>
## [RNN](#re_)
<div align=center><img src="https://github.com/FangChao1086/Machine_Learning/blob/master/依赖文件/RNN.jpg"></div>  

* T时刻，隐藏层神经元的激活值：
  * $$S_{t}=f\left(U * x_{t}+W * S_{t-1}+b_{1}\right)$$  
* T时刻，输出层的激活值：			
  * $$O_{t}=f\left(V * S_{t}+b_{2}\right)$$  


<span id="1x1卷积作用"></span>
## [1x1卷积作用](#re_)
* 实现跨通道的交互和信息整合
* 实现卷积核通道数的降维和升维
* 可以实现多个feature map的线性组合,而且可以实现与全连接层的等价效果。

<span id="Dropout"></span>
## [Dropout](#re_)
在神经网络训练的时候（前向传播），让神经单元以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征  
### Dropout的使用
* 训练
  * 使用Dropout
  * 以一定概率丢失神经元， 对每一个mini-batch，都使用新的Dropout
* 测试
  * 不使用Dropout，使用一个网络结构
    * 假设rate：P = 0.5
    * 若训练的 w_train = 1
    * 则测试时设置 w_test = w_train * (1 - p) =  0.5
  * 使用Dropout，集成学习训练很多的网络结构，然后平均
### dropout可以解决过拟合的原因：
1. 取平均的作用  
> 整个dropout过程就相当于对很多个不同的神经网络取平均。而不同的网络产生不同的过拟合，一些互为“反向”的拟合相互抵消就可以达到整体上减少过拟合  
2. 减少神经元之间复杂的共适应关系
