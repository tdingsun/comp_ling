from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import argparse
import unicodedata
import re
from collections import defaultdict
from tqdm import tqdm  # optional progress bar

PAD_TOKEN = "*PAD*"
START_TOKEN = "*START*"
UNK_TOKEN = "UNK"
STOP_TOKEN = "*STOP*"

class TranslationDataset(Dataset):
    def __init__(self, input_file, enc_seq_len, dec_seq_len,
                 bpe=True, target=None, word2id=None, flip=False):
        """
        Read and parse the translation dataset line by line. Make sure you
        separate them on tab to get the original sentence and the target
        sentence. You will need to adjust implementation details such as the
        vocabulary depending on whether the dataset is BPE or not.

        :param input_file: the data file pathname
        :param enc_seq_len: sequence length of encoder
        :param dec_seq_len: sequence length of decoder
        :param bpe: whether this is a Byte Pair Encoded dataset
        :param target: the tag for target language
        :param word2id: the word2id to append upon
        :param flip: whether to flip the ordering of the sentences in each line
        """
        self.enc_input_vectors = []
        self.dec_input_vectors = []
        self.label_vectors = []
        self.enc_input_lengths = []
        self.dec_input_lengths = []

        curr_id = enc_word2id[1]
        self.word2id = enc_word2id[0]

        if curr_id == 0:
            self.word2id = {PAD_TOKEN: 0, START_TOKEN: 1, STOP_TOKEN: 2}
            curr_id = 3

        self.target = target
        if target not in self.enc_word2id:
            self.word2id[target] = curr_id
            curr_id += 1

        # read the input file line by line and put the lines in a list.
        enc_lines, dec_lines = read_from_corpus(input_file)
        if flip:
            enc_lines, dec_lines = dec_lines, enc_lines

        # split the whole file (including both training and validation
        # data) into words and create the corresponding vocab dictionary.

        for line in enc_lines:
            enc_input_seq = [self.target] + line + [STOP_TOKEN]
            enc_input_vector = []
            for word in enc_input_seq:
                if word not in self.word2id:
                    self.word2id[word] = curr_id
                    curr_id += 1
                enc_input_vector.append(self.word2id[word])

            self.enc_input_lengths.append(len(enc_input_vector))
            self.enc_input_vectors.append(torch.tensor(enc_input_vector))

        for line in dec_lines:
            dec_input_seq = [START_TOKEN] + line + [STOP_TOKEN]
            dec_input_vector = []
            for word in dec_input_seq:
                if word not in self.word2id:
                    self.word2id[word] = curr_id
                    curr_id += 1
                dec_input_vector.append(self.word2id[word])

            label_vector = dec_input_vector[1:]

            self.dec_input_lengths.append(len(dec_input_vector))
            self.label_vectors.append(torch.tensor(label_vector))
            self.dec_input_vectors.append(torch.tensor(dec_input_vector))

  
        # create inputs and labels for both training and validation data
        # and make sure you pad your inputs.
        enc_first_pad = torch.zeros(enc_seq_len - len(self.enc_input_vectors[0]), dtype=torch.long)
        dec_first_pad = torch.zeros(dec_seq_len - len(self.dec_input_vectors[0]), dtype=torch.long)
        label_first_pad = torch.zeros(dec_seq_len - (len(self.dec_input_vectors[0]) - 1), dtype=torch.long)
        
        self.dec_input_vectors[0] = torch.cat((self.dec_input_vectors[0], dec_first_pad))
        self.enc_input_vectors[0] = torch.cat((self.enc_input_vectors[0], enc_first_pad))
        self.label_vectors[0] = torch.cat((self.label_vectors[0], label_first_pad))

        self.label_vectors = pad_sequence(self.label_vectors, batch_first=True)
        self.dec_input_vectors = pad_sequence(self.dec_input_vectors, batch_first=True)
        self.enc_input_vectors = pad_sequence(self.enc_input_vectors, batch_first=True)

        self.dec_input_lengths = torch.tensor(self.dec_input_lengths)
        self.enc_input_lengths = torch.tensor(self.enc_input_lengths)

        self.vocab_size = len(self.word2id)
        # Hint: remember to add start and pad to create inputs and labels

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        # Override method to return length of dataset
        return len(self.dec_input_vectors)

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index
        In this case, you should return your original and target sentence and
        anything else you may need in training/validation/testing.

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        # Override method to return the items in dataset
        item = {
            "enc_input_vector": self.enc_input_vectors[idx],
            "dec_input_vector": self.dec_input_vectors[idx],
            "label_vector": self.label_vectors[idx],
            "enc_input_lengths": self.enc_input_lengths[idx],
            "dec_input_lengths": self.dec_input_lengths[idx]
        }
        return item


def read_from_corpus(corpus_file):
    eng_lines = []
    frn_lines = []

    with open(corpus_file, 'r') as f:
        for line in f:
            words = line.split("\t")
            eng_line = words[0]
            frn_line = words[1][:-1]

            eng_line = re.sub('([.,!?():;])', r' \1 ', eng_line)
            eng_line = re.sub('\s{2,}', ' ', eng_line)
            frn_line = re.sub('([.,!?():;])', r' \1 ', frn_line)
            frn_line = re.sub('\s{2,}', ' ', frn_line)

            eng_lines.append(eng_line.split())
            frn_lines.append(frn_line.split())
    return eng_lines, frn_lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='path to BPE input')
    parser.add_argument('iterations', help='number of iterations', type=int)
    parser.add_argument('output_file', help='path to BPE output')
    args = parser.parse_args()
