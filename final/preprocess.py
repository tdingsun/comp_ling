from torch.utils.data import Dataset
import torch
import numpy as np
import random
import math

class MyDataset(Dataset):

    def __init__(self, input_fn, window_size, batch_size, word2id, char2id, max_word_len):
        super().__init__()
        self.word2id = {PAD_TOKEN: 0, MASK_TOKEN: 1} if word2id == None else word2id
        # Read data from file
        words_arr = read_file(input_fn) #array of all words
        contents = tokenize(words_arr, char2id, max_word_len) #breaking words into chars
        contents = contents[:len(contents) - (len(contents) % (window_size * batch_size))]
        # Split data into fixed length sequences
        self.sequences = [contents[i*window_size:(i+1)*window_size] for i in range(len(contents) // window_size)]
        self.sequences = torch.tensor(self.sequences)

        words_arr = [word2id[w] for w in words_arr[1:]] + [word2id[words_arr[-1]]] #array of all words converted to ids
        words_arr = words_arr[:len(words_arr) - (len(words_arr) % (window_size * batch_size))]
        self.labels = [words_arr[i*window_size:(i+1)*window_size] for i in range((len(words_arr)) // window_size)]
        self.labels = torch.tensor(self.labels)

        self.dataset_size = len(self.sequences)

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
    """
    Tokenizes words into char-ids

    Inputs:
    contents: A list of words
    char2id: Dictionary from characters to char-ids
    max_word_len: maximum word length

    Outputs:
    A list of lists of char-ids (each list represents a word)
    """
    tokenized_content = []
    for word in contents:
        vec = [char2id[char] for char in word]
        if len(vec) < max_word_len:
            vec += [char2id["*PAD*"] for _ in range(max_word_len - len(vec))]
        vec = [char2id["*BOW*"]] + vec + [char2id["*EOW*"]]
        tokenized_content.append(vec)
    return tokenized_content

def read_file(fname):
    """
    Reads files and splits and concatenates the lines

    Inputs:
    fname: filename

    Outputs: List of words (strings)
    """
    content = []
    with open(fname) as f:
        for line in f.readlines():
            content += line.strip().split()
    return content

def create_dicts(train_file, valid_file, test_file):
    """
    Creates word2id and char2id dicts

    input:
    train_file: filename for training file
    valid_file: filename for validation file
    test_file: filename for testing file

    Output:
    word2id: Dictionarie from words to word-ids,
    char2id: Dictionary from characters to char-ids
    """
    content = []
    content += read_file(train_file)
    content += read_file(valid_file)
    content += read_file(test_file)

    word2id = {}
    curr_id = 0
    for word in content:
        if word not in word2id:
            word2id[word] = curr_id
            curr_id += 1
            

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


