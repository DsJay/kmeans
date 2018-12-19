#!/usr/bin/python3
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import v_measure_score
import mnist_reader

def function1(x1):
    path1 = "./question2_cluster/"+str(x1)+".png"
    image1 = Image.open(path1)
    v = np.array(image1)
    return np.linalg.eig(v)[0]


a1 = function1(20)
a2 = function1(3)
a3 = function1(2)
a4 = function1(14)
a5 = function1(7)
a6 = function1(9)
a7 = function1(5)
a8 = function1(10)
a9 = function1(19)
a10 = function1(1)
alist = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]

num = range(0, 10)
array_sum = [0]*10
array_num = [0]*10
result = [0]*10000
flag = 0
reg = 40

while flag < reg:
    i = 1
    while i <= 10000:
        np_array = function1(i)
        t = 0
        for x in alist:
            num[t] = np.linalg.norm(np_array-x, ord=1)
            t += 1
        lable = num.index(min(num))
        #print i, lable
        result[i-1] = lable
        array_sum[lable] += np_array
        array_num[lable] += 1
        i = i+1
    m = 0
    while m < 10:
        alist[m] = array_sum[m]/array_num[m]
        m += 1
    flag += 1
    array_num.sort()
    print array_num
    array_sum = [0]*10
    array_num = [0]*10


data = pd.DataFrame({'lable': result})
data.to_csv('k-means_1202.csv', encoding='utf-8', index=False)

