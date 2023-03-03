import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Params:
    def __init__(self):
        self.batchsize = 512
        self.dense_layers = 3
        self.n_filters = 256
        self.filter_size = 24
        self.pooling_size = 15
        self.pooling_stride = 15
        self.dropout = 0.5
        self.dense_layer_size = 512
        self.lstm_out = 32
        
class bichrom_seq(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.params=params
        self.conv1d=nn.Conv1d(4, params.n_filters, params.filter_size)
        self.relu=nn.ReLU()
        self.batchNorm1d=nn.BatchNorm1d(params.n_filters)
        self.maxPool1d=nn.MaxPool1d(params.pooling_size,params.pooling_stride)
        self.lstm=nn.LSTM(params.n_filters, params.lstm_out, batch_first=True)
        self.model_dense_repeat = nn.Sequential()
        self.model_dense_repeat.append(nn.Linear(params.lstm_out, params.dense_layer_size))
        for idx in range(params.dense_layers):
            self.model_dense_repeat.append(nn.Linear(params.dense_layer_size, params.dense_layer_size))
            self.model_dense_repeat.append(nn.ReLU())
            self.model_dense_repeat.append(nn.Dropout(0.5))
        self.linear=nn.Linear(params.dense_layer_size, 1)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        xs=self.conv1d(x)
        xs=self.relu(xs)
        xs=self.batchNorm1d(xs)
        print(xs.shape)
        lin=xs.shape[-1]
        pad=((lin-1)*self.params.pooling_stride+1+(self.params.pooling_size-1)-lin)//2
        xs=F.pad(xs, (pad, pad), mode='constant', value=0)
        xs=self.maxPool1d(xs)
        print(xs.shape)
        xs=torch.permute(xs, (0, 2, 1))
        xs=self.lstm(xs)[0][:, -1, :]
        xs=self.model_dense_repeat(xs)
        xs=self.linear(xs)
        xs=self.sigmoid(xs)
        return xs

def build_seq_model(params):
    return bichrom_seq(params)