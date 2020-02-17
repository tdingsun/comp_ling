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
        # TODO: initialize embeddings, LSTM, and linear layers
        #encoder embedding layer
        
        #for BPE, theres just a single embedding layer because both use BPE joint vocab.

        if bpe:
            self.bpe_embedding = nn.Embedding(vocab_size, embedding_size)
        else:
            self.encoder_embedding = nn.Embedding(vocab_size, embedding_size)
            self.decoder_embedding = nn.Embedding(output_size, embedding_size)

        #encoder LSTM
        self.encoder_lstm = nn.LSTM(embedding_size, rnn_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(embedding_size, rnn_size, batch_first=True)
        #decoder embedding layer
        #decoder LSTM
        #dense layer
        self.dense = nn.Linear(rnn_size, vocab_size)

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
        # write forward propagation

        total_length = decoder_inputs.shape[1]

        # make sure you use pack_padded_sequence and pad_padded_sequence to
        # reduce calculation
        if self.bpe:
            enc_embeds = self.bpe_embedding(encoder_inputs)
            dec_embeds = self.bpe_embedding(decoder_inputs)
        else:
            enc_embeds = self.encoder_embedding(encoder_inputs)
            dec_embeds = self.decoder_embedding(decoder_inputs)

        #encode encoder inputs (encoder embedding layer)
        #pass those into LSTM layer
        packed_enc_embeds = pack_padded_sequence(enc_embeds, encoder_lengths, batch_first=True, enforce_sorted=False)
        _, enc_hc = self.encoder_lstm(packed_enc_embeds)

        #do same thing with decoder, pass initial state in as the output of the encoder input

        packed_dec_embeds = pack_padded_sequence(dec_embeds, decoder_lengths, batch_first=True, enforce_sorted=False)
        
        dec_out, _ = self.decoder_lstm(packed_dec_embeds, enc_hc)
        output, _ = pad_packed_sequence(dec_out, batch_first=True, total_length=total_length)
        
        #pass decoder output into dense
        logits = self.dense(output)
        return logits
