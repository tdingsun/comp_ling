from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, rnn_size, embedding_size, output_size,
                 enc_seq_len, dec_seq_len, bpe):
        """
        The Model class implements the LSTM-LM model.
        Feel free to initialize any variables that you find necessary in the
        constructor.

        :param vocab_size: The number of unique tokens in the input
        :param rnn_size: The size of hidden cells in LSTM/GRU
        :param embedding_size: The dimension of embedding space
        :param output_size: The vocab size in output sequence
        :param enc_seq_len: The sequence length of encoder
        :param dec_seq_len: The sequence length of decoder
        :param bpe: whether the data is Byte Pair Encoded (shares same vocab)
        """
        super().__init__()
        # TODO: initialize the vocab_size, rnn_size, embedding_size
        self.bpe = bpe
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        self.hidden_size = 512

        # TODO: initialize embeddings, LSTM, and linear layers
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.LSTM(embedding_size, rnn_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, rnn_size, batch_first=True)
        self.dense = nn.Linear(rnn_size, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, encoder_inputs, decoder_inputs, encoder_lengths,
                decoder_lengths):

        """
        Runs the forward pass of the model.

        :param inputs: word ids (tokens) of shape (batch_size, seq_len)
        :param encoder_lengths: array of actual lengths (no padding) encoder
                                inputs
        :param decoder_lengths: array of actual lengths (no padding) decoder
                                inputs

        :return: the logits, a tensor of shape
                 (batch_size, seq_len, vocab_size)
        """

        total_length = decoder_inputs.shape[1]

        enc_embeds = self.embedding(encoder_inputs)
        dec_embeds = self.embedding(decoder_inputs)

        packed_enc_embeds = pack_padded_sequence(enc_embeds, encoder_lengths, batch_first=True, enforce_sorted=False)
        _, enc_hc = self.encoder(packed_enc_embeds)

        packed_dec_embeds = pack_padded_sequence(dec_embeds, decoder_lengths, batch_first=True, enforce_sorted=False)
        dec_out, _ = self.decoder(packed_dec_embeds, enc_hc)
        output, _ = pad_packed_sequence(dec_out, batch_first=True, total_length=total_length)

        logits = self.dense(output)
        logits = self.dense2(logits)
        return logits
        # make sure you use pack_padded_sequence and pad_padded_sequence to
        # reduce calculation
