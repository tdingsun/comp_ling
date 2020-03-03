import torch
import torch.nn as nn
import numpy as np
import math, copy
from torch.nn import functional as F
from torch.autograd import Variable

#produce N identical layers
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Positional_Encoding_Layer(nn.Module):
    """
    Right now is fixed using sin and cos, I could try just doing a learned thing if it doesnt work.
    """
    def __init__(self, window_size, embedding_size, dropout):
        super(Positional_Encoding_Layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(window_size, embedding_size)
        position = torch.arange(0, window_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * -(math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)




def Self_Attention(K, V, Q, mask):
    """
	:param K: is [batch_size x window_size_keys x embedding_size]
	:param V: is [batch_size x window_size_values x embedding_size]
	:param Q: is [batch_size x window_size_queries x embedding_size]
	:return: attention
    """

    window_size_keys = K.shape[1]

    sqrt_k_dim = math.sqrt(window_size_keys)
    qk_t = torch.matmul(Q, K.transpose(-2, -1)) / sqrt_k_dim
    qk_t = qk_t.masked_fill(mask == 0, -1e9)

    attention = torch.matmul(F.softmax(qk_t, dim=-1), V)

    return attention

class Multi_Headed_Attention(nn.Module):
    def __init__(self, embedding_size, dropout, window_size):
        super(Multi_Headed_Attention, self).__init__()
        #make 8 different attention heads
        #split data for 8 different heads of size embedding_size/3
        #concatenate outputs of three heads
        #apply linear layer
        self.h = 8
        self.split_size = int(embedding_size / self.h)
        self.linear_v = nn.Linear(embedding_size, embedding_size)
        self.linear_k = nn.Linear(embedding_size, embedding_size)
        self.linear_q = nn.Linear(embedding_size, embedding_size)
        self.linear_end = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(p = dropout)

        pass
    def forward(self, inputs_k, inputs_v, inputs_q, mask):
        m = mask.unsqueeze(1)
        num_batches = inputs_k.size(0)
        k = self.linear_k(inputs_k).view(num_batches, -1, self.h, self.split_size).transpose(1, 2)
        v = self.linear_v(inputs_v).view(num_batches, -1, self.h, self.split_size).transpose(1, 2)
        q = self.linear_q(inputs_q).view(num_batches, -1, self.h, self.split_size).transpose(1, 2)

        x = Self_Attention(k, v, q, m) #add dropout or no?
        x = x.transpose(1, 2).contiguous().view(num_batches, -1, self.h * self.split_size)
        return self.linear_end(x)
    



class Feed_Forward(nn.Module):
    def __init__(self, embedding_size):
        super(Feed_Forward, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size
        self.layer_1 = nn.Linear(embedding_size, self.hidden_size)
        self.layer_2 = nn.Linear(embedding_size, self.hidden_size)

    def forward(self, x):
        out = self.layer_1(x)
        out = F.leaky_relu(out)
        out = self.layer_2(out)
        return out

class Encoder(nn.Module):
    """
        multihead, add+norm, feedforward, add+norm
    """
    def __init__(self, embedding_size, dropout, window_size):
        super(Encoder, self).__init__()
        self.multihead = Multi_Headed_Attention(embedding_size, dropout, window_size)
        self.ff_layer = Feed_Forward(embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, x, mask):
        atten_out = self.multihead(x, x, x, mask)
        atten_out += x
        atten_normalized = self.layer_norm(atten_out)

        ff_out = self.ff_layer(atten_normalized)
        ff_out += atten_normalized
        ff_norm = self.layer_norm(ff_out)

        return F.leaky_relu(ff_norm)

class Transformer(nn.Module):
    def __init__(self, vocab_size, window_size, batch_size, embedding_size):
        super(Transformer, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size
        self.dropout = 0.1

        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        #positional encoding layer
        self.positional_encoding_layer = Positional_Encoding_Layer(self.window_size, self.embedding_size, self.dropout)
        #encoder layer (transformer block) * 6
        self.encoder_layers = clones(Encoder(self.embedding_size, self.dropout, window_size), 6)
        #dense layers
        self.linear_1 = nn.Linear(self.embedding_size, self.hidden_size)
        self.linear_2 = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs, mask):
        embeddings = self.embedding(inputs)
        x = self.positional_encoding_layer(embeddings)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        x = self.linear_1(x)
        x = F.leaky_relu(x)
        x = self.linear_2(x)
        
        return x 