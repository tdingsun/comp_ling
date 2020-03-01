from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
import numpy as np
import unicodedata
import argparse
import re

PAD_TOKEN = "*PAD*"
START_TOKEN = "*START*"


def load_dataset(train_fn, test_fn, batch_size):
    """
    :param fn: filename for the dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for train and test
    :Comment: You don't have to shuffle the test dataset
    """
    word2id = {PAD_TOKEN: 0, START_TOKEN: 1}
    curr_id = 2

    train_lines = read_from_corpus(train_fn)
    test_lines = read_from_corpus(test_fn)
    
    max_train_len = len(max(train_lines, key = lambda i: len(i)))
    max_test_len = len(max(test_lines, key = lambda i: len(i)))
    max_seq_len = max(max_train_len, max_test_len)
    print("Window Size", max_seq_len)

    train_dataset = TransformerDataset(train_lines, word2id, curr_id, max_seq_len)
    test_dataset = TransformerDataset(test_lines, train_dataset.word2id, train_dataset.curr_id, max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return(train_loader, test_loader, len(test_dataset.word2id), max_seq_len)


class TransformerDataset(Dataset):
    def __init__(self, input_lines, word2id, curr_id, max_seq_len):
        """
        Read and parse the file line by line.
        Create a word2id vocab dict.
        Vectorize all data. 
        """
        self.input_vectors = []
        self.label_vectors = []
        self.lengths = []
        self.word2id = word2id
        self.curr_id = curr_id
        self.max_seq_len = max_seq_len
        self.vocab_size = 0

        for line in input_lines:
            label_vector = []
            for word in line:
                if word not in self.word2id:
                    self.word2id[word] = self.curr_id
                    self.curr_id += 1
                label_vector.append(self.word2id[word])
            input_vector = [self.word2id[START_TOKEN]] + label_vector[:-1]

            self.lengths.append(len(line))
            self.label_vectors.append(torch.tensor(label_vector))
            self.input_vectors.append(torch.tensor(input_vector))
        
        first_pad = torch.zeros(max_seq_len - len(self.input_vectors[0]), dtype=torch.long)
        self.input_vectors[0] = torch.cat((self.input_vectors[0], first_pad))
        self.label_vectors[0] = torch.cat((self.label_vectors[0], first_pad))

        self.input_vectors = pad_sequence(self.input_vectors, batch_first=True)
        self.label_vectors = pad_sequence(self.label_vectors, batch_first=True)

        self.lengths = torch.tensor(self.lengths)
        self.vocab_size = len(self.word2id)

    def __len__(self):
        return len(self.input_vectors)
    
    def __getitem__(self, idx):
        item = {
            "input_vectors": self.input_vectors[idx],
            "label_vectors": self.label_vectors[idx],
            "lengths": self.lengths[idx]
        }
        return item
    
        
def read_from_corpus(corpus_file):
    lines = []

    with open(corpus_file, 'r') as f:
        for line in f:
            seq = line.strip().split()
            lines.append(seq)
    return lines