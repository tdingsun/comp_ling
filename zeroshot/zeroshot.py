from comet_ml import Experiment
from preprocess import TranslationDataset, read_from_corpus
from model import Seq2Seq
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar

# TODO: Set hyperparameters
hyperparams = {
    "rnn_size": 128,  # assuming encoder and decoder use the same rnn_size
    "embedding_size": 128,
    "num_epochs": 1,
    "batch_size": 20,
    "learning_rate": 0.0015
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, experiment, hyperparams, bpe):
    """
    Trains the model.

    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    :param bpe: is bpe dataset or not
    """
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    model = model.train()
    with experiment.train():
        # training loop
        for e in range(hyperparams['num_epochs']):
            for batch in tqdm(train_loader):
                enc_inputs = batch['enc_input_vector'].to(device)
                dec_inputs = batch['dec_input_vector'].to(device)
                labels = batch['label_vector'].to(device)
                
                enc_lengths = batch['enc_input_lengths'].to(device)
                dec_lengths = batch['dec_input_lengths'].to(device)

                optimizer.zero_grad()
                y_pred = model(enc_inputs, dec_inputs, enc_lengths, dec_lengths)
                y_pred = torch.flatten(y_pred, 0, 1)
                y_actual = torch.flatten(labels, 0, 1)
                loss = loss_fn(y_pred, y_actual)
                loss.backward()
                optimizer.step()

def test(model, test_loader, experiment, hyperparams, bpe):
    """
    Validates the model performance as LM on never-seen data.

    :param model: the trained model to use for prediction
    :param test_loader: Dataloader of testing data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    :param bpe: is bpe dataset or not
    """
    # Define loss function, total loss, and total word count
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    word_count = 0
    total_wrong = 0

    model = model.eval()
    with experiment.test():
        for batch in tqdm(test_loader):
            enc_inputs = batch['enc_input_vector'].to(device)
            dec_inputs = batch['dec_input_vector'].to(device)
            labels = batch['label_vector'].to(device)
            
            enc_lengths = batch['enc_input_lengths'].to(device)
            dec_lengths = batch['dec_input_lengths'].to(device)
            y_pred = model(enc_inputs, dec_inputs, enc_lengths, dec_lengths)
            loss = loss_fn(torch.flatten(y_pred, 0, 1), torch.flatten(labels, 0, 1))
            num_words_in_batch = torch.sum(batch['dec_input_lengths']).item()
            total_loss += loss.item() * num_words_in_batch
            word_count += num_words_in_batch

            num_wrong = 0
            for i in range(len(dec_lengths)):
                diff = torch.argmax(y_pred[i, :], -1)[0:dec_lengths[i]] - labels[i, 0:dec_lengths[i]]
                num_wrong += np.count_nonzero(diff.cpu())
            total_wrong += num_wrong
        
        perplexity = np.exp(total_loss / word_count)
        accuracy = 1 - (total_wrong / word_count)

        print("perplexity:", perplexity)
        print("accuracy:", accuracy)
        experiment.log_metric("perplexity", perplexity)
        experiment.log_metric("accuracy", accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--corpus-files", nargs="*")
    parser.add_argument("-b", "--bpe", action="store_true",
                        help="use bpe data")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-m", "--multilingual-tags", nargs="*", default=[None],
                        help="target tags for translation")
    parser.add_argument("-z", "--zeroshot", action="store_true",
                        help="zeroshot translation")
    parser.add_argument("-r", "--rnnsize", nargs=1, default=64)
    parser.add_argument("-e", "--embedding", nargs=1, default=64)
    args = parser.parse_args()

    print(args.rnnsize)
    print(args.embedding)
    hyperparams["rnn_size"] = args.rnnsize
    hyperparams["embedding_size"] = args.embedding
    # Make sure you modify the `.comet.config` file
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    # TODO: Load dataset
    # Hint: Use random_split to split dataset into train and validate datasets
    # Hint: Use ConcatDataset to concatenate datasets
    # Hint: Make sure encoding and decoding lengths match for the datasets
    if args.zeroshot:
        data_tags = list(zip(args.corpus_files, args.multilingual_tags, [True, False, False, False]))
    else:
        data_tags = list(zip(args.corpus_files, args.multilingual_tags, [False, False]))
    
    #finding max encoder sequence length and decoder sequence length
    enc_seq_len = 0
    dec_seq_len = 0
    for input_file, tag, flip in data_tags:
        enc_lns, dec_lns = read_from_corpus(input_file)
        if flip:
            enc_lns, dec_lns = dec_lns, enc_lns
        dec_max = len(max(dec_lns, key = lambda i: len(i)))
        dec_seq_len = max(dec_seq_len, dec_max)
        enc_max = len(max(enc_lns, key = lambda i: len(i)))
        enc_seq_len = max(enc_seq_len, enc_max)
    enc_seq_len += 2
    dec_seq_len += 2

    #vocab stuff
    vocab_size = 0
    word2id = ({}, 0)
    datasets = []

    if args.zeroshot:
        #inputs would be dattrainED, dattrainEF, dattestED, dattestEF
        train_data = data_tags[0:2]
        test_data = data_tags[2:4]
        #dattrainED has to be flipped
        for input_file, tag, flip in train_data:
            d = TranslationDataset(input_file, enc_seq_len, dec_seq_len, args.bpe, tag, word2id, flip)
            word2id = (d.word2id, d.vocab_size)
            vocab_size = max(vocab_size, d.vocab_size)
            datasets.append(d)
        train_loader = DataLoader(ConcatDataset(datasets), batch_size=hyperparams['batch_size'], shuffle=True)

        #test files
        source_input = data_tags[2]
        target_input = data_tags[3]
        d = TranslationDataset(source_input[0], enc_seq_len, dec_seq_len, args.bpe, target_input[1], word2id, flip=False, target_input_file=target_input[0], ztest=True)
        word2id = (d.word2id, d.vocab_size)
        vocab_size = max(vocab_size, d.vocab_size)
        test_dataset = DataLoader(d, batch_size=hyperparams['batch_size'], shuffle=True)

    else:
        for input_file, tag, flip in data_tags:
            d = TranslationDataset(input_file, enc_seq_len, dec_seq_len, args.bpe, tag, word2id)
            word2id = (d.word2id, d.vocab_size)
            vocab_size = max(vocab_size, d.vocab_size)
            datasets.append(d)
        
        combined_dataset = ConcatDataset(datasets)
        test_size = len(combined_dataset) // 10
        train_size = len(combined_dataset) - test_size

        train_subset, test_subset = random_split(combined_dataset, (train_size, test_size))

        train_loader = DataLoader(train_subset, batch_size = hyperparams['batch_size'], shuffle=True)
        test_dataset = DataLoader(test_subset, batch_size = hyperparams['batch_size'], shuffle=True)

    model = Seq2Seq(
        vocab_size,
        hyperparams["rnn_size"],
        hyperparams["embedding_size"],
        vocab_size,
        enc_seq_len,
        dec_seq_len,
        args.bpe
    ).to(device)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("running training loop...")
        train(model, train_loader, experiment, hyperparams, args.bpe)
    if args.test:
        print("running testing loop...")
        test(model, test_dataset, experiment, hyperparams, args.bpe)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
