# -*- coding: utf-8 -*-
"""
@author: wangying

@date:2022.06.12
"""

import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import xlrd

train_num = 149
valid_num = 50

#input data

file = xlrd.open_workbook(r'F:\WY\PPBR\data\PPBR(0.3).xlsx')


train_ws = file.sheet_by_name('train')
valid_ws = file.sheet_by_name('valid')

def data(r1, r2, ws):
    y_ = []
    x_ = []
    for r in range(r1, r2):
        m = ws.cell(r, 10).value
        # print(m)
        #m = [m]
        m = m
        y_.append(m)
        x1_ = []
        for c in range(2, 10):
            n = ws.cell(r, c).value
            x1_.append(n)
        x_.append(x1_)

    return [x_, y_]

train_data = data(1, train_num + 1 , train_ws)
valid_data = data(1, valid_num + 1, valid_ws)

model = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=100, randam_state=42)
model.fit(train_data[0], train_data[1])

valid_predict = model.predict(valid_data[0])
train_predict = model.predict(train_data[0])

print("train_x:", train_data[0])
print("train_y:", train_data[1])
print("train_prey:", train_predict)

mae = mean_absolute_error(valid_data[1], valid_predict)
rmse = sqrt(mean_squared_error(valid_data[1], valid_predict))

# computational accuracy
a = []
print('total：', valid_num)

for i in range(0, valid_num):
     # print(y_pred[i])
     b = abs(valid_data[1] - valid_predict[i])
     #print(b)

     if b[i] <= 0.1:
         a.append(b[i])

print('mae', mae)
print('rmse', rmse)
print('Accurate number of predictions：', len(a))
accuracy = len(a) / valid_num
print('acc of Validation set：', accuracy)


