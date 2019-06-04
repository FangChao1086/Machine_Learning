# pandas
* [聚合](#聚合)
* [时间处理](#时间处理)

[链接：速查表_pandas](https://github.com/FangChao1086/machine_learning/tree/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E9%80%9F%E6%9F%A5%E8%A1%A8/Pandas)  
[参考链接：pandas加速-Modin](https://mp.weixin.qq.com/s/zLMa-fkvErXpQLWjcQbUzg)   
<span id="pandas编码objects_type"></span>
### pandas编码objects type
```python
# 编码所有的object type
from sklearn import preprocessing

for x in train.columns:
    if train[x].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[x].values))
        train[x] = lbl.transform(list(train[x].values))

```
* 排序
  * 特定类 数值计数 ：
  ```python
  data['A'].value_counts()
  ```
  
## 聚合  
* .agg
```python
'''
author:fangchao
date:2019/5/24

content:聚合
'''
import pandas as pd

dataset = pd.DataFrame({
    'user_id': [1, 2, 1],
    'coupon_id': [1, 2, 1]
})

t1 = dataset[['user_id', 'coupon_id']].copy()
t1['this_month_user_receive_same_coupon_count'] = 1
t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()

print(dataset)
print(t1)

'''
   user_id  coupon_id
0        1          1
1        2          2
2        1          1
   user_id  coupon_id  this_month_user_receive_same_coupon_count
0        1          1                                          2
1        2          2                                          1

'''
```

<span id="时间处理"></span>
## 时间处理
* pd.to_datetime
* pd.Timedelta
```python
def label(row):
    if row['Date_received'] == 'null':
        return -1
    if row['Date'] != 'null':
        td = pd.to_datetime(row['Date'], format='%Y%m%d') -  pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0
```
