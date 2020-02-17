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
                 bpe=True, target=None, word2id=None):
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
        """
        # TODO: read the input file line by line and put the lines in a list.
        # TODO: split the whole file (including both training and validation
        # data) into words and create the corresponding vocab dictionary.

        # TODO: create inputs and labels for both training and validation data
        #       and make sure you pad your inputs.
        self.enc_input_vectors = []
        self.dec_input_vectors = []
        self.label_vectors = []
        self.enc_input_lengths = []
        self.dec_input_lengths = []

        enc_curr_id = word2id[1]
        dec_curr_id = word2id[1]
        self.enc_word2id = word2id[0]
        self.dec_word2id = word2id[0]
        if enc_curr_id == 0:
            self.enc_word2id = {PAD_TOKEN: 0, START_TOKEN: 1, STOP_TOKEN: 2}
            self.dec_word2id = {PAD_TOKEN: 0, START_TOKEN: 1, STOP_TOKEN: 2}
            enc_curr_id = 3
            dec_curr_id = 3

        self.target = START_TOKEN
        if target != None:
            self.target = target
            if target not in self.enc_word2id:
                self.enc_word2id[target] = enc_curr_id
                enc_curr_id += 1

        if bpe:
            en_lines, fr_lines = read_from_corpus(input_file)
            for line in en_lines:
                label = line + [STOP_TOKEN]

                label_vector = []
                for word in label:
                    if word not in self.enc_word2id:
                        self.enc_word2id[word] = enc_curr_id
                        enc_curr_id += 1
                    label_vector.append(self.enc_word2id[word])
                dec_input_vector = [self.enc_word2id[self.target]] + label_vector[:-1]

                self.dec_input_lengths.append(len(dec_input_vector))
                self.label_vectors.append(torch.tensor(label_vector))
                self.dec_input_vectors.append(torch.tensor(dec_input_vector))

            for line in fr_lines:
                enc_input_seq = [self.target] + line

                enc_input_vector = []
                for word in enc_input_seq:
                    if word not in self.enc_word2id:
                        self.enc_word2id[word] = enc_curr_id
                        enc_curr_id += 1
                    enc_input_vector.append(self.enc_word2id[word])

                self.enc_input_lengths.append(len(enc_input_vector))
                self.enc_input_vectors.append(torch.tensor(enc_input_vector))


        else:
            en_lines, fr_lines = preprocess_vanilla(input_file)
            for line in en_lines:
                label = line + [STOP_TOKEN]

                label_vector = []
                for word in label:
                    if word not in self.dec_word2id:
                        self.dec_word2id[word] = dec_curr_id
                        dec_curr_id += 1
                    label_vector.append(self.dec_word2id[word])
                dec_input_vector = [self.dec_word2id[self.target]] + label_vector[:-1]

                self.dec_input_lengths.append(len(dec_input_vector))
                self.label_vectors.append(torch.tensor(label_vector))
                self.dec_input_vectors.append(torch.tensor(dec_input_vector))

            for line in fr_lines:
                enc_input_seq = [self.target] + line

                enc_input_vector = []
                for word in enc_input_seq:
                    if word not in self.enc_word2id:
                        self.enc_word2id[word] = enc_curr_id
                        enc_curr_id += 1
                    enc_input_vector.append(self.enc_word2id[word])

                self.enc_input_lengths.append(len(enc_input_vector))
                self.enc_input_vectors.append(torch.tensor(enc_input_vector))

        dec_first_pad = torch.zeros(dec_seq_len - len(self.dec_input_vectors[0]), dtype=torch.long)
        enc_first_pad = torch.zeros(enc_seq_len - len(self.enc_input_vectors[0]), dtype=torch.long)
        self.dec_input_vectors[0] = torch.cat((self.dec_input_vectors[0], dec_first_pad))
        self.enc_input_vectors[0] = torch.cat((self.enc_input_vectors[0],enc_first_pad))
        self.label_vectors[0] = torch.cat((self.label_vectors[0], dec_first_pad))
        self.label_vectors = pad_sequence(self.label_vectors, batch_first=True)
        self.dec_input_vectors = pad_sequence(self.dec_input_vectors, batch_first=True)
        self.enc_input_vectors = pad_sequence(self.enc_input_vectors, batch_first=True)
        self.dec_input_lengths = torch.tensor(self.dec_input_lengths)
        self.enc_input_lengths = torch.tensor(self.enc_input_lengths)
        #teacher forcing :  use english lines to make labels. 
        # Hint: remember to add start and pad to create inputs and labels
        self.enc_vocab_size = len(self.enc_word2id)
        self.dec_vocab_size = len(self.dec_word2id)

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


def learn_bpe(train_file, iterations):
    """
    learn_bpe learns the BPE from data in the train_file and return a
    dictionary of {Byte Pair Encoded vocabulary: count}.

    Note: The original vocabulary should not include '</w>' symbols.
    Note: Make sure you use unicodedata.normalize to normalize the strings when
          reading file inputs.

    You are allowed to add helpers.

    :param train_file: file of the original version
    :param iterations: number of iterations of BPE to perform

    :return: vocabulary dictionary learned using BPE
    """
    # Please implement the BPE algorithm.
    vocab = defaultdict(lambda: 0)
    with open(train_file, 'rt') as data_file:
        for line in data_file:
            l = unicodedata.normalize("NFKC", line)
            words = re.split(r'(\s+)', l.strip())
            for word in words:
                word = " ".join(word)
                vocab[word] += 1
    
    for i in tqdm(range(iterations)):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)

    return(vocab)

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_transforms(vocab):
    """
    get_transforms return a mapping from an unprocessed vocabulary to its Byte
    Pair Encoded counterpart.

    :param vocab: BPE vocabulary dictionary of {bpe-vocab: count}

    :return: dictionary of {original: bpe-version}
    """
    transforms = {}
    for vocab, count in vocab.items():
        word = vocab.replace(' ', '')
        bpe = vocab + " </w>"
        transforms[word] = bpe
    return transforms


def apply_bpe(train_file, bpe_file, vocab):
    """
    apply_bpe applies the BPE vocabulary learned from the train_file to itself
    and save it to bpe_file.

    :param train_file: file of the original version
    :param bpe_file: file to save the Byte Pair Encoded version
    :param vocab: vocabulary dictionary learned from learn_bpe
    """
    with open(train_file) as r, open(bpe_file, 'w') as w:
        transforms = get_transforms(vocab)
        for line in r:
            l = unicodedata.normalize("NFKC", line)
            words = re.split(r'(\s+)', l.strip())
            bpe_str = ""
            for word in words:
                if word.isspace():
                    bpe_str += word
                else:
                    bpe_str += transforms[word]
            bpe_str += "\n"
            w.write(bpe_str)


def count_vocabs(eng_lines, frn_lines):
    eng_vocab = defaultdict(lambda: 0)
    frn_vocab = defaultdict(lambda: 0)

    for eng_line in eng_lines:
        for eng_word in eng_line:
            eng_vocab[eng_word] += 1
    for frn_line in frn_lines:
        for frn_word in frn_line:
            frn_vocab[frn_word] += 1

    return eng_vocab, frn_vocab


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


def unk_words(eng_lines, frn_lines, eng_vocab, frn_vocab, threshold=5):
    for eng_line in eng_lines:
        for i in range(len(eng_line)):
            if eng_vocab[eng_line[i]] <= threshold:
                eng_line[i] = "UNK"

    for frn_line in frn_lines:
        for i in range(len(frn_line)):
            if frn_vocab[frn_line[i]] <= threshold:
                frn_line[i] = "UNK"


def preprocess_vanilla(corpus_file, threshold=5):
    """
    preprocess_vanilla unks the corpus and returns two lists of lists of words.

    :param corpus_file: file of the corpus
    :param threshold: threshold count to UNK
    """
    eng_lines, frn_lines = read_from_corpus(corpus_file)
    eng_vocab, frn_vocab = count_vocabs(eng_lines, frn_lines)
    unk_words(eng_lines, frn_lines, eng_vocab, frn_vocab, threshold)
    return eng_lines, frn_lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='path to BPE input')
    parser.add_argument('iterations', help='number of iterations', type=int)
    parser.add_argument('output_file', help='path to BPE output')
    args = parser.parse_args()

    vocab = learn_bpe(args.input_file, args.iterations)
    apply_bpe(args.input_file, args.output_file, vocab)
