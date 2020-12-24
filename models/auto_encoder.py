from .modules import *
import numpy as np

import torch
from torch import nn, optim


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):

        super(Encoder, self).__init__()
        self.init_batchnorm = TimeDistributed(
            nn.BatchNorm1d(input_size, momentum=0.01), batch_first=True
        )

        self.lstm_1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.lstm_2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x):
        x = self.init_batchnorm(x)
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, emb_size):
        super(AutoEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.encoder = Encoder(input_size, hidden_size)
        self.mapping_layer = TimeDistributed(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, emb_size),
                nn.ReLU(),
            )
        )
        self.decoder = Encoder(emb_size, input_size)

    def forward(self, x):
        x = self.encoder(x)
        x_map = self.mapping_layer(x)
        x = self.decoder(x_map)
        return x, x_map
