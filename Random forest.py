# -*- coding: utf-8 -*-
"""
@author: wangying

@date:2022.03.23
"""

import xlrd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

train_num = 149
valid_num = 149

# input data

file = xlrd.open_workbook(r'F:\WY\PPBR\data\PPBR(0.3).xlsx')
train_ws = file.sheet_by_name('train')
valid_ws = file.sheet_by_name('train')

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

# Random forest classifier
forest = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)
print(train_data[0])
print(train_data[1])
forest.fit(train_data[0], train_data[1])

y_pre1 = forest.predict(valid_data[0])

MSE = metrics.mean_squared_error(valid_data[1], y_pre1)
print(y_pre1)
print(MSE)

'''
rf1 = RandomForestClassifier()
rf2 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=3, random_state=0)
y_pre1 = rf1.predict(valid_data[0])
'''