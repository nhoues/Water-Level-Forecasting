import gc
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from .modules import *


class ConvTransformer(nn.Module):
    """
    Computes an encoder based on LSTM layers
    """

    def __init__(self, input_size, hidden_size):
        super(ConvTransformer, self).__init__()

        self.day_emb = nn.Embedding(7, 4)
        self.hour_emb = nn.Embedding(24, 12)

        self.init_batchnorm = TimeDistributed(nn.BatchNorm1d(input_size, momentum=0.01))

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size + 16,
                out_channels=hidden_size,
                kernel_size=2,
                padding=True,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.transformer = EncoderLayer(d_model=hidden_size, dropout=0.01)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, enc, day, hour):
        day = self.day_emb(day)
        hour = self.hour_emb(hour)
        enc = self.init_batchnorm(enc)
        enc = torch.cat([enc, hour, day], dim=2)
        enc = enc.permute(0, 2, 1)
        enc = self.conv(enc)
        enc = enc.permute(0, 2, 1)
        x_hat, _ = self.LSTM(enc)
        x_hat = self.fc(x_hat[:, -1, :])
        return x_hat
