from torch.utils.data import Dataset
import torch
import numpy as np
import random

PAD_TOKEN = '<PAD>'
MASK_TOKEN = '<MASK>'

class MyDataset(Dataset):
    # TODO: Create masked Penn Treebank dataset.
    #       You can change signature of the initializer.
    def __init__(self, input_fn, window_size, word2id=None):
        super().__init__()
        self.word2id = {PAD_TOKEN: 0, MASK_TOKEN: 1} if word2id == None else word2id
        # Read data from file
        contents = read_file(input_fn)
        contents = self.tokenize(contents)
        # Split data into fixed length sequences
        sentences = [contents[i:i+window_size] for i in range(len(contents) - window_size)]
        self.dataset_size = len(sentences)
        # Mask part of the data for BERT training
        self.inputs, self.outputs = self.random_masking(sentences)

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
            "input_vecs": self.inputs[i],
            "label_vecs": self.outputs[i]
        }
        return item

    def tokenize(self, contents):
        tokenized_content = []
        for word in contents:
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id) + 1
            tokenized_content.append(self.word2id[word])
        return tokenized_content
    
    def random_masking(self, sentences):
        outputs = []
        inputs = []
        for s in sentences:
            output_label = []
            input_vec = s

            for i, token in enumerate(s):
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15

                    if prob < 0.8: # 80% of the time
                        input_vec[i] = self.word2id[MASK_TOKEN]
                    elif prob < 0.9: # 10% of the time
                        input_vec[i] = random.randrange(len(self.word2id))
                    #else 10% stay the same
                    output_label.append(token)
                else:
                    output_label.append(self.word2id[PAD_TOKEN])

            outputs.append(output_label)
            inputs.append(input_vec)

        return torch.tensor(inputs), torch.tensor(outputs)



def read_file(fname):
    content = []
    with open(fname) as f:
        for line in f.readlines():
            content += line.lower().strip().split()
    return content
