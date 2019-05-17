'''
author:fangchao
date:2019/5/17

content:,leave_one_out, leave_p_out交叉验证
'''

import numpy
from sklearn.model_selection import LeaveOneOut, LeavePOut


def print_result(split_data):
    for train, test in split_data:
        output_train = ''
        output_test = ''

        bar = ["-"] * (len(train) + len(test))

        # Build our output for display from the resulting split
        for i in train:
            output_train = "{}({}: {}) ".format(output_train, i, data[i])

        for i in test:
            bar[i] = "T"
            output_test = "{}({}: {}) ".format(output_test, i, data[i])

        print("[ {} ]".format(" ".join(bar)))
        print("Train: {}".format(output_train))
        print("Test:  {}\n".format(output_test))


P_VAL = 2

data = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8]])

loocv = LeaveOneOut()
lpocv = LeavePOut(p=P_VAL)
split_loocv = loocv.split(data)
split_lpocv = lpocv.split(data)

print("Data:\n{}\n".format(data))
print("Leave-One-Out:\n")
print_result(split_loocv)
print("Leave-P-Out (where p = {}):\n".format(P_VAL))
print_result(split_lpocv)

'''
Data:
[[1 2]
 [3 4]
 [5 6]
 [7 8]]

Leave-One-Out:

[ T - - - ]
Train: (1: [3 4]) (2: [5 6]) (3: [7 8]) 
Test:  (0: [1 2]) 

[ - T - - ]
Train: (0: [1 2]) (2: [5 6]) (3: [7 8]) 
Test:  (1: [3 4]) 

[ - - T - ]
Train: (0: [1 2]) (1: [3 4]) (3: [7 8]) 
Test:  (2: [5 6]) 

[ - - - T ]
Train: (0: [1 2]) (1: [3 4]) (2: [5 6]) 
Test:  (3: [7 8]) 

Leave-P-Out (where p = 2):

[ T T - - ]
Train: (2: [5 6]) (3: [7 8]) 
Test:  (0: [1 2]) (1: [3 4]) 

[ T - T - ]
Train: (1: [3 4]) (3: [7 8]) 
Test:  (0: [1 2]) (2: [5 6]) 

[ T - - T ]
Train: (1: [3 4]) (2: [5 6]) 
Test:  (0: [1 2]) (3: [7 8]) 

[ - T T - ]
Train: (0: [1 2]) (3: [7 8]) 
Test:  (1: [3 4]) (2: [5 6]) 

[ - T - T ]
Train: (0: [1 2]) (2: [5 6]) 
Test:  (1: [3 4]) (3: [7 8]) 

[ - - T T ]
Train: (0: [1 2]) (1: [3 4]) 
Test:  (2: [5 6]) (3: [7 8]) 
'''
