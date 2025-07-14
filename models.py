# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 19:21:01 2025

@author: sinam
"""

import math
import torch
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm, Linear, ReLU

import torch.nn as nn
import torch.nn.functional as F





################ CNN Backbone

def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 5,padding='same'),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )



class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = conv_block(6, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.conv5 = conv_block(512, 1024)

        self.global_pool = nn.AdaptiveAvgPool1d((1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x=self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x









################ Transformer Backbone (Code taken from https://github.com/dl4sits/BreizhCrops)

class PETransformerModel(nn.Module):
    def __init__(self, input_dim=6, num_classes=3, d_model=64, n_head=2, n_layers=5,
                 d_inner=128, activation="relu", dropout=0.017998950510888446, max_len=28):

        super(PETransformerModel, self).__init__()
        self.modelname = f"PeTransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
                         f"dropout={dropout}"

        encoder_layer = TransformerEncoderLayer(d_model, n_head, d_inner, dropout, activation)
        encoder_norm = LayerNorm(d_model)

        self.inlinear = Linear(input_dim, d_model)
        self.relu = ReLU()
        self.transformerencoder = TransformerEncoder(encoder_layer, n_layers, encoder_norm)
        self.flatten = Flatten()
        self.pe = PositionalEncoding(d_model, max_len=max_len)

        """
        self.sequential = Sequential(
            ,
            ,
            ,
            ,
            ReLU(),

        )
        """

    def forward(self,x):
        x = self.inlinear(x)
        x = self.pe(x)
        x = x.transpose(0, 1) # N x T x D -> T x N x D
        x = self.transformerencoder(x)
        x = x.transpose(0, 1) # T x N x D -> N x T x D
        x = x.max(1)[0]
        x = self.relu(x)

        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)





################ LSTM Backbone (Code taken from https://github.com/Lab-IDEAS/DeepCropMapping)

class DCM(nn.Module):
    def __init__(
        self, input_feature_size=6,hidden_size=256,num_layers=2,bidirectional=True,dropout=0.5,num_classes=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )  # i/o: (batch, seq_len, num_directions*input_/hidden_size)
        num_directions = 2 if bidirectional else 1
        self.attention = nn.Linear(
            in_features=num_directions * hidden_size,
            out_features=1,
        )
        self.fc = nn.Linear(
            in_features=num_directions * hidden_size,
            out_features=num_classes,
        )


    def forward(self, x):
        self.lstm.flatten_parameters()
        # lstm_out: (batch, seq_len, num_directions*hidden_size)
        lstm_out, _ = self.lstm(x)
        # softmax along seq_len axis
        attn_weights = F.softmax(F.relu(self.attention(lstm_out)), dim=1)
        # attn (after permutation): (batch, 1, seq_len)
        fc_in = attn_weights.permute(0, 2, 1).bmm(lstm_out)
        #fc_out = self.fc(fc_in)
        return fc_in, attn_weights.squeeze()


################ Fully connected network

class FC(nn.Module):
    def __init__(self,input_dim):
        super(FC, self).__init__()
        self.fco = nn.Linear(input_dim, 3)
    def forward(self, x):
        x=self.fco(x)
        return x
