'''
author:fangchao
date:2019/5/18

content:决策树
'''

from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import itertools
import random

classes = {
    'supplies': ['low', 'med', 'high'],
    'weather': ['raining', 'cloudy', 'sunny'],
    'worked?': ['yes', 'no']
}
data = [
    ['low', 'sunny', 'yes'],
    ['high', 'sunny', 'yes'],
    ['med', 'cloudy', 'yes'],
    ['low', 'raining', 'yes'],
    ['low', 'cloudy', 'no'],
    ['high', 'sunny', 'no'],
    ['high', 'raining', 'no'],
    ['med', 'cloudy', 'yes'],
    ['low', 'raining', 'yes'],
    ['low', 'raining', 'no'],
    ['med', 'sunny', 'no'],
    ['high', 'sunny', 'yes']
]
target = ['yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no']

# 转化为onehot矩阵
categories = [classes['supplies'], classes['weather'], classes['worked?']]
encoder = OneHotEncoder(categories=categories)
x_data = encoder.fit_transform(data)  # 稀疏矩阵（a,b） c :坐标点，坐标点的值，其他处为0

# 模型
model = DecisionTreeClassifier()
tree = model.fit(x_data, target)

# 获取随机数据，进行预测； 预测结果（随初始数据有关）：['yes' 'no' 'no' 'no' 'yes']
predict_data = []
for _ in itertools.repeat(None, 5):
    predict_data.append([
        random.choice(classes['supplies']),
        random.choice(classes['weather']),
        random.choice(classes['worked?'])
    ])
predict_result = tree.predict(encoder.transform(predict_data))

# 输出树的结构, 保存树的结构为pdf文件
feature_name = (
        ['supplies-' + x for x in classes['supplies']] +
        ['weather-' + x for x in classes['weather']] +
        ['worked?-' + x for x in classes['worked?']]
)
dot_data = export_graphviz(tree, filled=True, proportion=True, feature_names=feature_name)
graph = graphviz.Source(dot_data)
graph.render(filename='decision_tree', cleanup=True, view=True)


# 输出数据
def format_array(arr):
    return "".join(["| {:<10}".format(item) for item in arr])


def print_table(data, results):
    line = "day  " + format_array(list(classes.keys()) + ["went shopping?"])
    print("-" * len(line))
    print(line)
    print("-" * len(line))

    for day, row in enumerate(data):
        print("{:<5}".format(day + 1) + format_array(row + [results[day]]))
    print("")


print("Training Data:")
print_table(data, target)
print("Predicted Random Results:")
print_table(predict_data, predict_result)

'''
Training Data:
---------------------------------------------------------
day  | supplies  | weather   | worked?   | went shopping?
---------------------------------------------------------
1    | low       | sunny     | yes       | yes       
2    | high      | sunny     | yes       | no        
3    | med       | cloudy    | yes       | no        
4    | low       | raining   | yes       | no        
5    | low       | cloudy    | no        | yes       
6    | high      | sunny     | no        | no        
7    | high      | raining   | no        | no        
8    | med       | cloudy    | yes       | no        
9    | low       | raining   | yes       | no        
10   | low       | raining   | no        | yes       
11   | med       | sunny     | no        | yes       
12   | high      | sunny     | yes       | no        

Predicted Random Results:
---------------------------------------------------------
day  | supplies  | weather   | worked?   | went shopping?
---------------------------------------------------------
1    | med       | cloudy    | yes       | no        
2    | low       | raining   | no        | yes       
3    | low       | raining   | yes       | no        
4    | med       | sunny     | no        | yes       
5    | med       | raining   | no        | yes  
'''
