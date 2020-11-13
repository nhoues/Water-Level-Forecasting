import gc
import numpy as np

import torch
from torch import nn 
import torch.nn.functional as F

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y
    
class StackedLSTMs(nn.Module):
    """
    Computes an encoder based on LSTM layers
    """

    def __init__(self,input_size,hidden_size):
        super(StackedLSTMs, self).__init__()
        
        self.day_emb  = nn.Embedding(7, 4)
        self.hour_emb = nn.Embedding(24, 12)
        
        self.init_batchnorm = TimeDistributed(nn.BatchNorm1d(input_size, momentum=0.01))
        
        self.LSTM1 = nn.LSTM(
            input_size=input_size + 16 , hidden_size=hidden_size, batch_first=True, bidirectional=True
        )
        
        self.LSTM2 = nn.LSTM(
            input_size=hidden_size*2, hidden_size=hidden_size, batch_first=True, bidirectional=True
        )
        
        self.fc = nn.Sequential(
                            nn.Linear(2*hidden_size , 64),
                            nn.ReLU(inplace=True),
                            nn.Linear(64, 32),
                            nn.ReLU(inplace=True),
                            nn.Linear(32, 1),
                        )
        
    def forward(self, enc , day , hour ):
        day  = self.day_emb(day)
        hour = self.hour_emb(hour)
        enc = self.init_batchnorm(enc)
        enc = torch.cat([enc,hour,day] , dim = 2 )
        x_hat, _ = self.LSTM1(enc)
        x_hat, _ = self.LSTM2(x_hat)
        x_hat = self.fc(x_hat[:,-1,:])
        return x_hat