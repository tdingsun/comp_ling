from comet_ml import Experiment
from preprocess import *
from model import *
from torch.utils.data import DataLoader, random_split
from torch import nn

import torch
import numpy as np
import argparse
from tqdm import tqdm

hyper_params = {
    "batch_size": 20,
    "num_epochs": 3,
    "learning_rate": 0.001,
    "embedding_size": 66
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

experiment = Experiment(project_name="transformer")
experiment.log_parameters(hyper_params)

# Train the Model
def train(model, train_loader, experiment, hyperparams):
    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["learning_rate"])

    model = model.train()
    with experiment.train():
        for e in range(hyper_params['num_epochs']):
            for batch in tqdm(train_loader):
                x = batch['input_vectors'].to(device)
                y = batch['label_vectors'].to(device)
                lengths = batch['lengths'].to(device)
                optimizer.zero_grad()
                y_pred = model(x)

                loss = loss_fn(torch.flatten(y_pred, 0, 1), torch.flatten(y, 0, 1))
                loss.backward()
                optimizer.step()

                num_words_in_batch = torch.sum(lengths).item()
                total_batch_loss = loss.item()*num_words_in_batch
                perplexity = np.exp(total_batch_loss / num_words_in_batch)
                experiment.log_metric("perplexity", perplexity)

                num_wrong_in_batch = 0
                for i in range(len(lengths)):
                    
                    diff = torch.argmax(y_pred[i, :], -1)[0:lengths[i]] - y[i, 0:lengths[i]]
                    num_wrong_in_batch += np.count_nonzero(diff.cpu())

                accuracy = 1 - (num_wrong_in_batch / num_words_in_batch)
                experiment.log_metric("accuracy", accuracy)
        # Forward + Backward + Optimize

        # Compute train accuracy

        # Log perplexity to Comet.ml using experiment.log_metric


# Test the Model
def test(model, train_loader, experiment, hyperparams):
    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    word_count = 0
    total_wrong = 0

    model = model.eval()
    with experiment.test():
        for batch in tqdm(test_loader):
            x = batch['input_vectors'].to(device)
            y = batch['label_vectors'].to(device)
            lengths = batch['lengths'].to(device)

            y_pred = model(x)
            loss = loss_fn(torch.flatten(y_pred, 0, 1), torch.flatten(y, 0, 1))

            num_words_in_batch = torch.sum(lengths).item()
            total_loss += loss.item()*num_words_in_batch
            word_count += num_words_in_batch

            num_wrong_in_batch = 0
            for i in range(len(lengths)):
                diff = torch.argmax(y_pred[i, :], -1)[0:lengths[i]] - y[i, 0:lengths[i]]
                num_wrong_in_batch += np.count_nonzero(diff.cpu())
            total_wrong += num_wrong_in_batch
        
        perplexity = np.exp(total_loss / word_count)
        accuracy = 1 - (total_wrong / word_count)
        print("perplexity: ", perplexity)
        print("accuracy: ", accuracy)
        experiment.log_metric("perplexity", perplexity)
        experiment.log_metric("accuracy", accuracy)

        # Log perplexity to Comet.ml using experiment.log_metric


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
    args = parser.parse_args()
    
    # Data Loader (Input Pipeline)
    train_loader, test_loader, vocab_size, window_size = load_dataset(args.train_file, args.test_file, hyper_params['batch_size'])
  
    # Initialize your transformer using the hyper-parameters
    model = Transformer(
        vocab_size,
        window_size,
        hyper_params['batch_size'],
        hyper_params['embedding_size']
    ).to(device)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("running training loop...")
        train(model, train_loader, experiment, hyper_params)
    if args.test:
        print("running testing loop...")
        test(model, test_loader, experiment, hyper_params)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')

