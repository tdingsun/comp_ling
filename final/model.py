from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from torch.autograd import Variable
import math

class Highway(nn.Module):
    """Highway network"""
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, input_size, bias=True)

    def forward(self, x):
        t = torch.sigmoid(self.fc1(x))
        return torch.mul(t, F.relu(self.fc2(x))) + torch.mul(1-t, x)


class CharLM(nn.Module):
    def __init__(self, char_e_dim, word_e_dim, vocab_size, char_vocab_size, seq_len, batch_size):
        super().__init__()
        # TODO: Initialize BERT modules
        self.char_e_dim = char_e_dim
        self.word_e_dim = word_e_dim
        self.vocab_size = vocab_size
        self.char_vocab_size = char_vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.cnn_batch_size = self.seq_len * self.batch_size

        self.highway_input_dim = 1100

        self.char_embedding_layer = nn.Embedding(self.char_vocab_size, self.char_e_dim)
        self.cnn_w1 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(self.char_e_dim, 1), bias=True)
        self.cnn_w2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(self.char_e_dim, 2), bias=True)
        self.cnn_w3 = nn.Conv2d(in_channels=1, out_channels=150, kernel_size=(self.char_e_dim, 3), bias=True)
        self.cnn_w4 = nn.Conv2d(in_channels=1, out_channels=200, kernel_size=(self.char_e_dim, 4), bias=True)
        self.cnn_w5 = nn.Conv2d(in_channels=1, out_channels=200, kernel_size=(self.char_e_dim, 5), bias=True)
        self.cnn_w6 = nn.Conv2d(in_channels=1, out_channels=200, kernel_size=(self.char_e_dim, 6), bias=True)
        self.cnn_w7 = nn.Conv2d(in_channels=1, out_channels=200, kernel_size=(self.char_e_dim, 7), bias=True)

        self.convolutions = [self.cnn_w1, self.cnn_w2, self.cnn_w3, self.cnn_w4, self.cnn_w5, self.cnn_w6, self.cnn_w7]
        #tanh activation
        #max-over-time pooling for all of them, then concat. 
        self.batch_norm = nn.BatchNorm1d(self.highway_input_dim, affine=False)
        #highway
        self.highway1 = Highway(self.highway_input_dim)
        self.highway2 = Highway(self.highway_input_dim)

        #lstm
        self.lstm = nn.LSTM(input_size=self.highway_input_dim, hidden_size=self.word_e_dim, num_layers=2, bias=True, dropout=0.5, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.word_e_dim, self.vocab_size)


    def forward(self, x, hidden, generate=False):
        #input: batchsize x seq_len x max_word_len+2
        if generate:
            self.batch_size = x.size()[0]
            self.seq_len = x.size()[1]
            self.cnn_batch_size = self.batch_size * self.seq_len
        x = x.contiguous().view(-1, x.size()[2])
        #batch_size*seq_len x max_word_len+2
        x = self.char_embedding_layer(x) #output: batch_size*seq_len x max_wrd_len+2 x char_emb_dim (15)
        x = torch.transpose(x.view(x.size()[0], 1, x.size()[1], -1), 2, 3) #output: batch_size*seq_len x 1 x max_word_len+2 x char_emb_dim
        x = self.conv_layers(x) #output: batch_size*seq_len x total_num_filters (525)
        x = self.batch_norm(x) #output: batch_size*seq_len x total_num_filters (525)
        x = self.highway1(x) #output: batch_size*seq_len x total_num_filters (525)
        x = self.highway2(x) #output: batch_size*seq_len x total_num_filters (525)
        x = x.contiguous().view(self.batch_size, self.seq_len, -1) #output: batch_size x seq_len x total_num_filters (525)
        x, hidden = self.lstm(x, hidden) #output: batch_size x seq_len x lstm hidden size (300)
        x = self.dropout(x)
        x = x.contiguous().view(self.cnn_batch_size, -1) #output: batch_size*seq_len x lstm hidden size (300)
        x = self.linear(x) #output: batch_size*seq_len x vocab size
        return x, hidden

    def conv_layers(self, x):
        chosen_list = list()
        for conv in self.convolutions:
            feature_map = torch.tanh(conv(x))
            # (batch_size, out_channel, 1, max_word_len-width+1)
            chosen = torch.max(feature_map, 3)[0]
            # (batch_size, out_channel, 1)            
            chosen = chosen.squeeze()
            # (batch_size, out_channel)
            chosen_list.append(chosen)
        
        # (batch_size, total_num_filers)
        return torch.cat(chosen_list, 1)
    
    def getEmbedding(x):
        x = x.contiguous().view(-1, x.size()[2])
        #batch_size*seq_len x max_word_len+2
        x = self.char_embedding_layer(x) #output: batch_size*seq_len x max_wrd_len+2 x char_emb_dim (15)
        x = torch.transpose(x.view(x.size()[0], 1, x.size()[1], -1), 2, 3) #output: batch_size*seq_len x 1 x max_word_len+2 x char_emb_dim
        x = self.conv_layers(x) #output: batch_size*seq_len x total_num_filters (525)
        x = self.batch_norm(x) #output: batch_size*seq_len x total_num_filters (525)
        x = self.highway1(x) #output: batch_size*seq_len x total_num_filters (525)
        x = self.highway2(x) #output: batch_size*seq_len x total_num_filters (525)
        return x
    

    def getWordFromEmbedding(x):
        #input: batch_size*seq_len x total_num_filters (525)
        x = self.linear(x) #output: batch_size*seq_len x vocab size
        return x
    
