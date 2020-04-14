from comet_ml import Experiment
from data import MyDataset, read_file
from model import BERT
from embedding_analysis import embedding_analysis
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar

# TODO: Set hyperparameters
hyperparams = {
    "num_epochs": 40,
    "batch_size": 128,
    "lr": 1e-4,
    "seq_len": 64
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, loss_fn, optimizer, word2vec, experiment, hyperparams):
    """
    Training loop that trains BERT model.

    Inputs:
    - model: A BERT model
    - train_loader: Dataloader of training data
    - experiment: comet.ml experiment object
    - hyperparams: Hyperparameters dictionary
    """
    batch_id = 0
    model = model.train()
    with experiment.train():
        for batch in tqdm(train_loader):
            x = batch['input_vecs'].to(device)
            y = batch['label_vecs'].to(device)
            mask = y != 0 #true where not zero, flase everywhere else

            optimizer.zero_grad()

            y_pred = model(x, mask)
            loss = loss_fn(torch.flatten(y_pred, 0, 1), torch.flatten(y, 0, 1))
            loss.backward()
            optimizer.step()

            if batch_id % 1500 == 0:
                print(loss.item())
                torch.save(model.state_dict(), './model.pt')
            batch_id += 1

def test(model, test_loader, loss_fn, word2vec, experiment, hyperparams):
    """
    Testing loop for BERT model and logs perplexity and accuracy to comet.ml.

    Inputs:
    - model: A BERT model
    - train_loader: Dataloader of training data
    - experiment: comet.ml experiment object
    - hyperparams: Hyperparameters dictionary
    """
    model = model.eval()
    total_loss = 0
    word_count = 0
    total_wrong = 0
    with experiment.test(), torch.no_grad():
        for batch in tqdm(test_loader):
            x = batch['input_vecs'].to(device)
            y = batch['label_vecs'].to(device)
            mask = y != 0 #true where not zero, flase everywhere else

            y_pred = model(x, mask)
            loss = loss_fn(torch.flatten(y_pred, 0, 1), torch.flatten(y, 0, 1))
            print(loss.item())

            num_words_in_batch = torch.nonzero(y).size(0)
            total_loss += loss.item()*num_words_in_batch
            word_count += num_words_in_batch

            pred = torch.argmax(y_pred, -1)
            pred = torch.where(y == 0, y, pred)
            diff = pred - y
            total_wrong += np.count_nonzero(diff.cpu())
            
        perplexity = np.exp(total_loss / word_count)
        accuracy = 1 - (total_wrong / word_count)
        print(perplexity)
        print(accuracy)
        experiment.log_metric("perplexity", perplexity)
        experiment.log_metric("accuracy", accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-a", "--analysis", action="store_true",
                        help="run embedding analysis")
    args = parser.parse_args()

    # TODO: Make sure you modify the `.comet.config` file
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    # TODO: Load dataset
    train_set = MyDataset(args.train_file, hyperparams['seq_len'])
    test_set = MyDataset(args.test_file, hyperparams['seq_len'], word2id=train_set.word2id)
    word2vec = test_set.word2id
    train_loader = DataLoader(train_set, batch_size=hyperparams['batch_size'], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=hyperparams['batch_size'], shuffle=False)
    num_tokens = len(word2vec)

    model = BERT(hyperparams["seq_len"], num_tokens, n=2).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index = 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])

    if args.load:
        print("loading model")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        for e in range(hyperparams['num_epochs']):
            train(model, train_loader, loss_fn, optimizer, word2vec, experiment, hyperparams)
            test(model, test_loader, loss_fn, word2vec, experiment, hyperparams)
    if args.test:
        test(model, test_loader, loss_fn, word2vec, experiment, hyperparams)
    if args.save:
        torch.save(model.state_dict(), './model.pt')
    if args.analysis:
        embedding_analysis(model, experiment, train_set, test_set,
                           hyperparams["batch_size"], word2vec, device)
