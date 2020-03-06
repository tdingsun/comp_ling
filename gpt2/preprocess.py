from transformers import *
from torch.utils.data import Dataset, DataLoader
import torch

def load_transformer_dataset(train_fn, test_fn, tokenizer, batch_size):
    """
    :param fn: filename for the dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for train and test
    :Comment: This preprocess step is different from the previous ones. In this assignment, we are interested in using a pre-trained model.
    So, we have to use the exact vocabulary the pre-trained model was trained with. We are using the GPT-2 model, so pass your data through
    the GPT-2 tokenizer to get the word ids. You don't need to create your own dictionary.
    """

    # print(tokenizer.bos_token)
    # print(tokenizer.bos_token_id)
    # print(tokenizer.eos_token)
    # print(tokenizer.eos_token_id)
    # print(tokenizer.pad_token)
    # print(tokenizer.pad_token_id)

    train_inputs = []
    train_labels = []
    train_lengths = []
    with open(train_fn, 'r') as f:
        for line in f:
            label = line.strip().split()
            input_seq = [tokenizer.bos_token] + label[:-1]
            train_lengths.append(len(input_seq))
            train_inputs.append(torch.tensor(tokenizer.encode(input_seq, pad_to_max_length=True)))
            train_labels.append(torch.tensor(tokenizer.encode(label, pad_to_max_length=True)))
    train_lengths = torch.tensor(train_lengths)

    test_inputs = []
    test_labels = []
    test_lengths = []
    with open(test_fn, 'r') as f:
        for line in f:
            label = line.strip().split()
            input_seq = [tokenizer.bos_token] + label[:-1]
            test_lengths.append(len(input_seq))
            test_inputs.append(torch.tensor(tokenizer.encode(input_seq, pad_to_max_length=True)))
            test_labels.append(torch.tensor(tokenizer.encode(label, pad_to_max_length=True)))
    test_lengths = torch.tensor(test_lengths)

    train_dataset = TransformerDataset(train_inputs, train_labels, train_lengths)
    test_dataset = TransformerDataset(test_inputs, test_labels, test_lengths)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, tokenizer.pad_token_id + 1, 1024

def load_gpt2_dataset(test_fn, tokenizer, batch_size):
    test_inputs = []
    test_labels = []
    test_lengths = []
    with open(test_fn, 'r') as f:
        for line in f:
            label = line.strip().split()
            input_seq = [tokenizer.bos_token] + label[:-1]
            test_lengths.append(len(input_seq))
            test_inputs.append(torch.tensor(tokenizer.encode(input_seq)))
            test_labels.append(torch.tensor(tokenizer.encode(label)))
    test_lengths = torch.tensor(test_lengths)
    test_inputs = torch.cat(test_inputs).unsqueeze(1)
    print(test_inputs.shape)
    test_labels = torch.cat(test_labels).unsqueeze(1)
    print(test_labels.shape)
    test_dataset = GPT2Dataset(test_inputs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader

        
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
