# -*- coding: utf-8 -*-
"""
@author: wangying

@date:2022.03.24
"""

import xlrd
from sklearn import neighbors
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score # R square
import numpy as np
#train_num = 453
#valid_num = 151

train_num = 149
valid_num = 50  # 0.25

# input data

file = xlrd.open_workbook(r'F:\WY\PPBR\data\PPBR（0.3）.xlsx')



train_ws = file.sheet_by_name('train')
valid_ws = file.sheet_by_name('test')

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

rmse_val = []  # to store rmse values for different k
mae_val = []
acc_val = []
for K in range(79):
    K = K + 1
    print('K = ', K)
    model = neighbors.KNeighborsRegressor(n_neighbors=K)

    model.fit(train_data[0], train_data[1])  # fit the model
    pred = np.ndarray.tolist(model.predict(valid_data[0]))  # make prediction on test set
    error = sqrt(mean_squared_error(valid_data[1], pred))  # calculate rmse
    rmse_val.append(error)  # store rmse values
    print('RMSE value is:', error)
    print('pred', pred)

    # 计算准确度
    a = []
    print('total：', valid_num)
    for i in range(0, valid_num):
        # print(y_pred[i])
        b = abs(valid_data[1][i] - pred[i])
        # print(b)

        if b <= 0.1:
            a.append(b)

    mae = mean_absolute_error(valid_data[1], pred)
    mae_val.append(mae)
    rmse = sqrt(mean_squared_error(valid_data[1], pred))
    R2 = r2_score(valid_data[1], pred)
    print('Accurate number of predictions：', len(a))
    accuracy = len(a) / valid_num
    acc_val.append(accuracy)
    print('acc of Validation set：', accuracy)
    print('mae', mae)
    print('rmse', rmse)
    print('r2', R2)
    print('***********')

curve = pd.DataFrame(mae_val) #elbow curve
#curve = pd.DataFrame(acc_val) #elbow curve
curve.plot()
plt.show()
