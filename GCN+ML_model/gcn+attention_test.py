# -*- coding: utf-8 -*-
"""
@author: wangying

@date:2022.08.20
"""


from rdkit import Chem
import numpy as np
from rdkit.Chem import Draw
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch.nn as nn
import torch.nn.functional as F
import deepchem as dc
import torch
from sklearn import neighbors
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score # R square

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

import xlrd
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module




max_natoms = 457
test_num = 49

file = xlrd.open_workbook(r'F:\WY\OBA\oba_0.25.xlsx')


ws = file.sheet_by_name('test')

Y = []
S = []
for r in range(1, test_num + 1):
    y = ws.cell(r, 2).value
    y = [y]
    Y.append(y)

    s = ws.cell(r, 1).value
    s = s
    S.append(s)

file.release_resources()
del file

class MolDataset(Dataset):
    def __init__(self, smiles, properties, max_natoms, normalize_A=False):
        self.smiles = smiles
        self.properties = properties
        self.max_natoms = max_natoms

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        s = self.smiles[idx]
        m = Chem.MolFromSmiles(s)
        natoms = m.GetNumAtoms()
        #print(idx)
        #print(s)
        #print(natoms)

        #A = GetAdjacencyMatrix(m) + np.eye(natoms)
        A = GetAdjacencyMatrix(m)

        A_padding = np.zeros((self.max_natoms, self.max_natoms))
        A_padding[:natoms, :natoms] = A

        I = np.eye(A_padding.shape[0])
        A_hat = A_padding + I
        D_hat = np.array(np.sum(A_hat, axis=0))
        D_hat = np.power(D_hat, -0.5)  # !!!!
        D_hat = np.diag(D_hat)  # A_hat的度矩阵
        # D_hat1 = np.linalg.inv(D_hat)
        # A = np.dot(D_hat1, A_hat)
        A1 = np.dot(D_hat, A_hat)
        A = np.dot(A1, D_hat)


        X = [self.atom_feature(m, i) for i in range(natoms)]
        for i in range(natoms, max_natoms):
            X.append(np.zeros(78))
        X = np.array(X)

        sample = dict()
        sample['X'] = torch.from_numpy(X)
        sample['A'] = torch.from_numpy(A)
        sample['Y'] = torch.from_numpy(np.array(self.properties[idx]))
        #print(sample)

        return sample



    # one-hot
    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        # print list((map(lambda s: x == s, allowable_set)))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def atom_feature(self, m, atom_i):

        atom = m.GetAtomWithIdx(atom_i)

        return np.array((self.one_of_k_encoding_unk(atom.GetSymbol(),
                                               ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                                'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                                'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',  # H?
                                                'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                                'Cr', 'Pt', 'Hg', 'Pb', 'other']) +  # 'C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P', 'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other'
                         self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                         self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                         self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                         [atom.GetIsAromatic()]), dtype=float)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
#attention
class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        #self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)!!!
        self.attn_dropout = nn.Dropout(0.0)  #!!!

        # self-attention
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #support = torch.mm(input, self.weight)
        support = torch.matmul(input, self.weight)

        #output = torch.spmm(adj, support)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):

        super(GCN, self).__init__()

        self.embedding = nn.Linear(78, nfeat)
        self.att1 = SelfAttention(4, 100, 100, 0.99)
        self.gc1 = GraphConvolution(nfeat, 50)
        self.att2 = SelfAttention(2, 50, 50, 0.99)
        self.gc2 = GraphConvolution(50, 20)
        self.att3 = SelfAttention(1, nhid, nhid, 0.99)
        self.gc3 = GraphConvolution(20, 10)
        self.fc = nn.Linear(10, 1)
        self.dropout = dropout


    def forward(self, x, adj):
        x = self.embedding(x)
        x = torch.einsum('ijk,ikl->ijl', (adj, x))
        x = F.relu(self.att1(x))
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.att2(x))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.att3(x))
        x = F.relu(self.gc3(x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(x)
        # x = torch.sum(x, dim=1)
        x = x.mean(1)
        retval = self.fc(x)

        return retval

test_dataset = MolDataset(S, Y, max_natoms)

test_dataloader = DataLoader(test_dataset, batch_size=test_num, num_workers=0, shuffle=True)




model = GCN(100, 20, 0.9).cuda()
model.load_state_dict(torch.load('epoch_12_model.pth'))

loss_fn = nn.MSELoss()
loss_list = []

accuracy2 = 0
val_bestacc = 0

for i_batch, batch in enumerate(test_dataloader):
    x, y, A = \
        batch['X'].cuda().float(), batch['Y'].cuda().float(), batch['A'].cuda().float()

    pred = model(x, A).squeeze(-1)

    y1 = y.cpu().numpy()

    pred1 = pred.detach().cpu().numpy()
    print(y1)
    print(pred1)

    Pred = []
    for i in range(len(pred1)):
        pred2 = [pred1[i]]
        Pred.append(pred2)



    mae = mean_absolute_error(y1, Pred)
    mse = mean_squared_error(y1, Pred)
    rmse = sqrt(mean_squared_error(y1, Pred))
    R2 = r2_score(y1, Pred)



    a = []
    b = abs(Pred-y1)



    for i in range(len(b)):
        if b[i] <= 0.1:
            a.append(b[i])

    l = len(a)

    accuracy1 = float(l) / len(b)
    accuracy1 = ('%.2f' %accuracy1)

    print('test_acc:', accuracy1)
    print('test_mae', mae)
    print('test_rmse', rmse)
    print('test_R2', R2)



'''
plt.scatter(y1, Pred, s=20)
plt.plot([0, 1], [0, 1])
plt.xlabel('True')
plt.ylabel('Pred')
plt.show()
'''


x1 = [m for m in range(1, test_num + 1)]
'''
plt.scatter(x1, Pred, marker='v', label='pred_y')
plt.scatter(x1, y1, marker='o', label='y')'''


plt.figure(figsize=(4, 3))

plt.scatter(x1, Pred, marker='v')
plt.scatter(x1, y1, marker='o')

plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right', fontsize=5, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
              ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
plt.show()
