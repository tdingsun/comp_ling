from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from torch.autograd import Variable
import math


class Positional_Encoding_Layer(nn.Module):
    """
    Right now is fixed using sin and cos, I could try just doing a learned thing if it doesnt work.
    """
    def __init__(self, window_size, embedding_size):
        super(Positional_Encoding_Layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(window_size, embedding_size).float()
        position = torch.arange(0, window_size).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * -(math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + Variable(self.pe[:, :x.size(1)], requires_grad=False)


class BERT(nn.Module):
    def __init__(self, seq_len, num_words, d_model=768, h=12, n=12):
        super().__init__()
        # TODO: Initialize BERT modules
        self.seq_len = seq_len
        self.num_words = num_words
        self.d_model = d_model
        self.h = h
        self.n = n

        self.embedding_layer = nn.Embedding(num_words, self.d_model)
        self.positional_encoding_layer = Positional_Encoding_Layer(self.seq_len, self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.h, dim_feedforward=(self.d_model*4))
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n)

        self.linear = nn.Linear(self.d_model, self.num_words)

    def forward(self, x):
        # TODO: Write feed-forward step
        embeddings = self.embedding_layer(x)
        out = self.positional_encoding_layer(embeddings)
        out = self.transformer_encoder(out)
        out = self.linear(out)
        return out

    def get_embeddings(self, x):
        # TODO: Write function that returns BERT embeddings of a sequence
        pass
