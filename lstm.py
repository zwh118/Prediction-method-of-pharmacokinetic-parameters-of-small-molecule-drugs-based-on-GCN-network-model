# -*- coding: utf-8 -*-
"""
@author: wangying

@date:2022.05.05
"""

from rdkit import Chem
import numpy as np
import scipy.sparse as sp
from rdkit.Chem import Draw
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score # R square
from math import sqrt
#from pygcn.layers import GraphConvolution
import deepchem as dc
import torch

from torch.utils.data import Dataset, DataLoader

import xlrd
import openpyxl
import os

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


train_num = 453
valid_num = 152

# input data

file = xlrd.open_workbook(r'F:\WY\PPBR\data\ppbr.xlsx')
train_ws = file.sheet_by_name('train')
valid_ws = file.sheet_by_name('test')

def data(r1, r2, ws):
    y_ = []
    x_ = []
    for r in range(r1, r2):
        m = ws.cell(r, 10).value
        # print(m)
        #m = [m]
        y_.append(m)
        x1_ = []
        for c in range(2, 10):
            n = ws.cell(r, c).value
            x1_.append(n)
        x_.append(x1_)

    return [x_, y_]


# LSTM
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, output_size=1, hidden_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.forwardCalculation = nn.Linear(hidden_size, output_size)  # dense layer

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        print('x:', x)
        print('_:', _)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        #print('x:', x)
        return x

device = torch.device("cpu")

if (torch.cuda.is_available()):
    device = torch.device("cuda:0")
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')



train_data = data(1, train_num + 1, train_ws)
valid_data = data(1, valid_num + 1, valid_ws)

INPUT_FEATURES_NUM = 5
OUTPUT_FEATURES_NUM = 1
#print('y_:', train_data[1])
train_data_x = np.array(train_data[0]).astype('float32')

train_data_y = np.array(train_data[1]).astype('float32')


#print('train_data_x', train_data_x)
#print('train_data_y', train_data_y)

#train_x_tensor = train_data_x.reshape(-1, 8, INPUT_FEATURES_NUM)  # set batch size to 1

#print('train_x_tensor', train_x_tensor)
#print('train_x_tensor', np.shape(train_x_tensor))

train_y_tensor = train_data_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 1
#print('train_y_tensor', train_y_tensor)
#print('train_y_tensor', np.shape(train_y_tensor))

# transfer data to pytorch tensor
#train_x_tensor = torch.from_numpy(train_x_tensor)
train_y_tensor = torch.from_numpy(train_y_tensor)
#print('train_y_tensor', train_y_tensor)
#print('train_y_tensor', np.shape(train_y_tensor))



# train model
lr = 0.005
model = LstmRNN(78, 1, num_layers=1).cuda()

for param in model.parameters():
    if param.dim() == 1:
        continue
        nn.init.constant(param, 0)
    else:
        nn.init.xavier_normal_(param)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)

loss_fn = nn.MSELoss()

print('loss', loss_fn)