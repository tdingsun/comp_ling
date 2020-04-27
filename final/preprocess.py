from torch.utils.data import Dataset
import torch
import numpy as np
import random
import math

class MyDataset(Dataset):
    # TODO: Create masked Penn Treebank dataset.
    #       You can change signature of the initializer.
    def __init__(self, input_fn, window_size, word2id, char2id, max_word_len):
        super().__init__()
        self.word2id = {PAD_TOKEN: 0, MASK_TOKEN: 1} if word2id == None else word2id
        # Read data from file
        words_arr = read_file(input_fn) #array of all words
        contents = tokenize(words_arr, char2id, max_word_len) #breaking words into chars
        # Split data into fixed length sequences
        self.sequences = [contents[i*window_size:(i+1)*window_size] for i in range((len(contents) - 1) // window_size)]
        self.sequences = torch.tensor(self.sequences)
        # sentences = [contents[i:i+window_size] for i in range(len(contents) - window_size)]
        print("time for labels")
        words_arr = words_arr[1:]
        self.labels = [words_arr[i*window_size:(i+1)*window_size] for i in range((len(contents) - 1) // window_size)]
        # self.labels = [word2id[w] for w in words_arr[1:]] + [word2id[words_arr[-1]]]
        print(self.label)
        print("make torch")

        self.labels = torch.tensor(self.labels)
        print("LABELLLSS")
        print(self.labels.shape)
        self.dataset_size = len(self.sequences)
        # Mask part of the data for BERT training

    def __len__(self):
        """
        __len__ should return a the length of the dataset

        :return: an integer length of the dataset
        """
        return self.dataset_size

    def __getitem__(self, i):
        """
        __getitem__ should return a tuple or dictionary of the data at some
        index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        # TODO: Override method to return the ith item in dataset
        item = {
            "input_vecs": self.sequences[i],
            "label_vecs": self.labels[i]
        }
        return item

def tokenize(contents, char2id, max_word_len):
    tokenized_content = []
    for word in contents:
        vec = [char2id[char] for char in word]
        if len(vec) < max_word_len:
            vec += [char2id["*PAD*"] for _ in range(max_word_len - len(vec))]
        vec = [char2id["*BOW*"]] + vec + [char2id["*EOW*"]]
        tokenized_content.append(vec)
    return tokenized_content

def read_file(fname):
    content = []
    with open(fname) as f:
        for line in f.readlines():
            content += line.lower().strip().split()
    return content

def create_dicts(train_file, valid_file, test_file):
    content = []
    content += read_file(train_file)
    content += read_file(valid_file)
    content += read_file(test_file)

    word2id = {word:id for id, word in enumerate(set(content))}

    char2id = {}
    curr_id = 1
    for word in word2id:
        for char in word:
            if char not in char2id:
                char2id[char] = curr_id
                curr_id += 1

    char2id["*BOW*"] = len(char2id) + 1
    char2id["*EOW*"] = len(char2id) + 1
    char2id["*PAD*"] = 0

    return word2id, char2id


