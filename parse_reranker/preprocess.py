from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

PAD_TOKEN = "*PAD*"
START_TOKEN = "*START*"

class ParsingDataset(Dataset):
    def __init__(self, input_file):
        """
        Read and parse the train file line by line. Create a vocabulary
        dictionary that maps all the unique tokens from your data as
        keys to a unique integer value. Then vectorize your
        data based on your vocabulary dictionary.

        :param input_file: the data file pathname
        """

        self.input_sentences = []
        self.label_sentences = []
        self.input_vectors = []
        self.label_vectors = []
        self.lengths = []
        self.word2id = {PAD_TOKEN: 0, START_TOKEN: 1}
        self.curr_id = 2
        self.max_seq_len = 0
        self.vocab_size = 0
        
        # read the input file line by line and put the lines in a list.
        with open(input_file, 'rt', encoding='latin') as data_file:
            for line in data_file: 
                # create inputs and labels for both training and validation data
                # and make sure you pad your inputs.
                label = line.strip().split()
                input_seq = [START_TOKEN] + label[:-1]
                
                label_vectorized = []
                # split the whole file (including both training and validation
                # data) into words and create the corresponding vocab dictionary.
                for word in label:
                    if word not in self.word2id:
                        self.word2id[word] = self.curr_id
                        self.curr_id += 1
                    label_vectorized.append(self.word2id[word])
                input_vectorized = [self.word2id[START_TOKEN]] + label_vectorized

                self.lengths.append(len(input_seq))
                self.max_seq_len = max(self.max_seq_len, len(input_seq))
                self.label_sentences.append(label)
                self.input_sentences.append(input_seq)
                self.label_vectors.append(torch.tensor(label_vectorized))
                self.input_vectors.append(torch.tensor(input_vectorized))

            self.input_vectors = pad_sequence(self.input_vectors, batch_first=True)
            self.label_vectors = pad_sequence(self.label_vectors, batch_first=True)

        self.vocab_size = len(self.word2id)
        print(self.max_seq_len)

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        # Override method to return length of dataset
        return len(self.input_vectors)

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        # Override method to return the items in dataset
        item = {
            # "input_sentence": self.input_sentences[idx],
            "input_vector": self.input_vectors[idx],
            # "label_sentence": self.label_sentences[idx],
            "label_vector": self.label_vectors[idx],
            "lengths": self.lengths[idx]
        }
        return item

class RerankingDataset(Dataset):
    def __init__(self, parse_file, gold_file, word2id):
        """
        Read and parse the parse files line by line. Unk all words that has not
        been encountered before (not in word2id). Split the data according to
        gold file. Calculate number of constituents from the gold file.

        :param parse_fxile: the file containing potential parses
        :param gold_file: the file containing the right parsings
        :param word2id: the previous mapping (dictionary) from word to its word
                        id
        """
        #parse_file: reranker_train
        #gold file: gold
        #test parses: conv
        pass

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        # TODO: Override method to return length of dataset
        pass

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        # TODO: Override method to return the items in dataset
        pass
