from transformers import *
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

def load_transformer_dataset(train_fn, test_fn, tokenizer, batch_size):
    """
    :param fn: filename for the dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for train and test
    :Comment: This preprocess step is different from the previous ones. In this assignment, we are interested in using a pre-trained model.
    So, we have to use the exact vocabulary the pre-trained model was trained with. We are using the GPT-2 model, so pass your data through
    the GPT-2 tokenizer to get the word ids. You don't need to create your own dictionary.
    """

    max_seq_len = get_max_seq_len(train_fn, test_fn) + 1

    train_inputs = []
    train_labels = []
    train_lengths = []
    with open(train_fn, 'r') as f:
        for line in f:
            train_lengths.append(len(line))
            encoded_line = tokenizer.encode(line)
            input_seq = [tokenizer.bos_token_id] + encoded_line[:-1]
            label = encoded_line
            train_inputs.append(torch.tensor(input_seq))
            train_labels.append(torch.tensor(label))
    train_lengths = torch.tensor(train_lengths)

    test_inputs = []
    test_labels = []
    test_lengths = []
    with open(test_fn, 'r') as f:
        for line in f:
            test_lengths.append(len(line))
            encoded_line = tokenizer.encode(line)
            input_seq = [tokenizer.bos_token_id] + encoded_line[:-1]
            label = encoded_line
            test_inputs.append(torch.tensor(input_seq))
            test_labels.append(torch.tensor(label))
    test_lengths = torch.tensor(test_lengths)

    # train_inputs  = pad_sequence(train_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    # test_inputs  = pad_sequence(test_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    # train_labels  = pad_sequence(train_labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    # test_labels  = pad_sequence(test_labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    train_dataset = TransformerDataset(train_inputs, train_labels, train_lengths)
    test_dataset = TransformerDataset(test_inputs, test_labels, test_lengths)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, tokenizer.pad_token_id + 1, 1024

def load_gpt2_dataset(test_fn, tokenizer, batch_size):
    test_inputs = []
    test_labels = []
    with open(test_fn, 'r') as f:
        for line in f:
            input_seq = tokenizer.bos_token + " " + line
            label = line + " " + tokenizer.eos_token
            test_inputs.append(torch.tensor(tokenizer.encode(input_seq)))
            test_labels.append(torch.tensor(tokenizer.encode(label)))
    # test_inputs = torch.cat(test_inputs).unsqueeze(1)
    # print(test_inputs.shape)
    # test_labels = torch.cat(test_labels).unsqueeze(1)
    # print(test_labels.shape)
    test_dataset = GPT2Dataset(test_inputs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader

def get_max_seq_len(train_fn, test_fn):
    train_lines = []
    test_lines = []
    with open(train_fn, 'r') as f:
        for line in f:
            train_lines.append(line.strip().split())
    with open(test_fn, 'r') as f:
        for line in f:
            test_lines.append(line.strip().split())
    
    max_train_len = len(max(train_lines, key = lambda i: len(i)))
    max_test_len = len(max(test_lines, key = lambda i: len(i)))
    max_seq_len = max(max_train_len, max_test_len)

    return max_seq_len




        
class TransformerDataset(Dataset):
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


class GPT2Dataset(Dataset):
    def __init__(self, inputs, labels):
        self.input_vectors = inputs
        self.label_vectors = labels
    
    def __len__(self):
        return len(self.input_vectors)
    def __getitem__(self, idx):
        item = {
            "input_vectors": self.input_vectors[idx],
            "label_vectors": self.label_vectors[idx],
        }
        return item
