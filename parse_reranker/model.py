from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class LSTMLM(nn.Module):
    def __init__(self, vocab_size, rnn_size, embedding_size):
        """
        The Model class implements the LSTM-LM model.
        Feel free to initialize any variables that you find necessary in the
        constructor.

        :param vocab_size: The number of unique tokens in the data
        :param rnn_size: The size of hidden cells in LSTM/GRU
        :param embedding_size: The dimension of embedding space
        """
        super().__init__()
        # TODO: initialize the vocab_size, rnn_size, embedding_size
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        # TODO: initialize embeddings, LSTM, and linear layers
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, rnn_size, batch_first=True)

        self.dense = nn.Linear(rnn_size, vocab_size)
    def forward(self, inputs, lengths):

        """
        Runs the forward pass of the model.

        :param inputs: word ids (tokens) of shape (window_size, batch_size)
        :param lengths: array of actual lengths (no padding) of each input

        :return: the logits, a tensor of shape
                 (window_size, batch_size, vocab_size)
        """
        # TODO: write forward propagation
        total_length = inputs.shape[1]

        embeds = self.embeddings(inputs) #output size: batchsize x max_seq_len x embedding_len
        packed_input = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=total_length, padding_value=0)
        logits = self.dense(output)
        return logits #batch size, window size, vocab size
        # make sure you use pack_padded_sequence and pad_padded_sequence to
        # reduce calculation
