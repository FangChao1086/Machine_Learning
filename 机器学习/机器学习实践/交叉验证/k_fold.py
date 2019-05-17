'''
author:fangchao
date:2019/5/17

content:k_fold交叉验证
'''

import numpy as np
from sklearn.model_selection import KFold

num_split = 3

data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

split_data = KFold(n_splits=num_split).split(data)

print('原始数据data:\n', data)
for train, test in split_data:
    print('...\n训练')
    for i in train:
        print('output_train', i, data[i])
    print('测试')
    for i in test:
        print('output_test', i, data[i])

'''
原始数据data:
 [[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 7  8]
 [ 9 10]
 [11 12]]
...
训练
output_train 2 [5 6]
output_train 3 [7 8]
output_train 4 [ 9 10]
output_train 5 [11 12]
测试
output_test 0 [1 2]
output_test 1 [3 4]
...
训练
output_train 0 [1 2]
output_train 1 [3 4]
output_train 4 [ 9 10]
output_train 5 [11 12]
测试
output_test 2 [5 6]
output_test 3 [7 8]
...
训练
output_train 0 [1 2]
output_train 1 [3 4]
output_train 2 [5 6]
output_train 3 [7 8]
测试
output_test 4 [ 9 10]
output_test 5 [11 12]
'''
