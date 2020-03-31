from transformers import *
from torch.utils.data import Dataset, DataLoader
import torch
import re


def load_dataset(train_fn, test_fn, tokenizer, batch_size, window_size):
    """
    :param fn: filename for the dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for
    train and test

    :Comment: This function should be similary as the last GPT-2 assignemnent.
    We are still using the GPT-2 tokenizer to get the word ids.
    One thing to note is that when preprocessing the dataset, please exclude
    all the sentences defining the speaker's persona, in this assignment,
    we will just be training and testing on the chat text data. Also for each
    record of chat in the format of "sentence \t response", add BOS and EOS
    token to the start and end it, also add a special token between sentence
    and response to differentiate them from each other.
    """
    train_loader = read_files(train_fn, tokenizer, window_size, batch_size)
    test_loader = read_files(test_fn, tokenizer, window_size, batch_size)

    return train_loader, test_loader


def read_files(fn, tokenizer, max_len, batch_size):
    inputs = []
    labels = []
    lengths = []

    with open(fn, 'r') as f:
        for line in f:
            #exclude any sentences starting with  your persona: or partner's persona
            #only sentences wtih sentence tab response
            #BOS, sentence, special token, repsonse, EOS
            if " persona: " not in line:
                #TODO: remove numbers? (sentence ids)
                sen_res = line.replace('\t', tokenizer.sep_token)
                sen_res = re.sub('\d', '', sen_res).strip()
                inpt = tokenizer.bos_token + sen_res
                print(inpt)
                label = sen_res + tokenizer.eos_token
                encoded_line = tokenizer.encode(inpt, max_length = max_len)
                lengths.append(len(encoded_line))
                inputs.append(torch.tensor(tokenizer.encode(inpt, max_length=max_len, pad_to_max_length=True)))
                labels.append(torch.tensor(tokenizer.encode(label, max_length=max_len, pad_to_max_length=True)))

    dataset = ChatDataset(inputs, labels, lengths)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader



class ChatDataset(Dataset):
    def __init__(self, inputs, labels, lengths):
        self.input_vectors = inputs
        self.label_vectors = labels
        self.lengths = lengths
    
    def __len__(self):
        return len(self.input_vectors)

    def __getitem__(self, idx):
        item = {
            "input_vectors": self.input_vectors[idx],
            "label_vectors": self.label_vectors[idx],
            "lengths": self.lengths[idx]
        }
        return item

                



