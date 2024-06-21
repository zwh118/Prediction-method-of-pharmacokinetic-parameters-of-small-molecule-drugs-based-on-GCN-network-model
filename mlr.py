# -*- coding: utf-8 -*-
"""
@author: wangying

@date:2022.03.25
"""

import xlrd
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

train_num = 149
valid_num = 50

# input data

file = xlrd.open_workbook(r'F:\WY\PPBR\data\PPBR(0.3).xlsx')
train_ws = file.sheet_by_name('train')
valid_ws = file.sheet_by_name('valid')

def data(r1, r2, ws):
    y_ = []
    x_ = []
    for r in range(r1, r2):
        m = ws.cell(r, 10).value
        # print(m)
        m = [m]
        y_.append(m)
        x1_ = []
        for c in range(2, 10):
            n = ws.cell(r, c).value
            x1_.append(n)
        x_.append(x1_)

    return [x_, y_]

train_data = data(1, train_num + 1 , train_ws)
valid_data = data(1, valid_num + 1, valid_ws)

mlr = linear_model.LinearRegression()
mlr.fit(train_data[0], train_data[1])

y_pred = mlr.predict(valid_data[0])
print(y_pred)

# accuracy in computation
a = []
print('total：', valid_num)

for i in range(0, valid_num):
     # print(y_pred[i])
     b = abs(valid_data[1] - y_pred[i])
     #print(b)

     if b[i] <= 0.1:
         a.append(b[i])

mae = mean_absolute_error(valid_data[1], y_pred)
rmse = sqrt(mean_squared_error(valid_data[1], y_pred))

print('mae', mae)
print('rmse', rmse)
print('Accurate number of predictions：', len(a))
accuracy = len(a) / valid_num
print('acc of Validation set：', accuracy)
print('***********')