from comet_ml import Experiment
from preprocess import ParsingDataset, RerankingDataset
from model import LSTMLM
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar

# TODO: Set hyperparameters
hyperparams = {
    "rnn_size": 64,
    "embedding_size": 64,
    "num_epochs": 1,
    "batch_size": 20,
    "learning_rate": 0.001
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train(model, train_loader, experiment, hyperparams):
    """
    Trains the model.

    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    """
    # TODO: Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # TODO: Write training loop
    model = model.train()
    with experiment.train():
        for e in range(hyperparams["num_epochs"]):
            for batch in tqdm(train_loader):
                x = batch['input_vector'].to(device)
                y = batch['label_vector'].to(device)
                lengths = batch['lengths'].to(device)
                optimizer.zero_grad()
                y_pred = model(x, lengths)
                y_pred = torch.flatten(y_pred, 0, 1)
                y_actual = torch.flatten(y, 0, 1)
                loss = loss_fn(y_pred,  y_actual)
                loss.backward()
                optimizer.step()



def validate(model, validate_loader, experiment, hyperparams):
    """
    Validates the model performance as LM on never-seen data.

    :param model: the trained model to use for prediction
    :param validate_loader: Dataloader of validation data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    """
    # TODO: Define loss function, total loss, and total word count
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    word_count = 0
    

    # Write validating loop
    model = model.eval()
    with experiment.validate():
        for batch in tqdm(validate_loader):
            x = batch['input_vector'].to(device)
            y = batch['label_vector'].to(device)
            lengths = batch['lengths'].to(device)

            y_pred = model(x, lengths)
            print(y.shape)
            print(y_pred.shape)
            y_pred = torch.flatten(y_pred, 0, 1)
            y_actual = torch.flatten(y, 0, 1)
      
            loss = loss_fn(y_pred,  y_actual)

            # TODO: loss is correct, but I think im calculating perplexity wrong
            total_loss += loss.item()
            word_count += torch.sum(batch['lengths']).item()
        perplexity = np.exp(total_loss / word_count)
        print("perplexity:", perplexity)
        experiment.log_metric("perplexity", perplexity)


def test(model, test_dataset, experiment, hyperparams):
    """
    Validates and tests the model for parse reranking.

    :param model: the trained model to use for prediction
    :param test_dataset: Dataloader of testing data
    :param experiment: comet.ml experiment object
    :param hyperparams: Hyperparameters dictionary
    """
    softmax_fn = torch.nn.Softmax(dim=2)
    # TODO: Write testing loops
    model = model.eval()
    with experiment.test():
        gold_acc = 0
        correct_acc = 0
        total_acc = 0
        for batch in tqdm(test_dataset):
            gold_total_tags = batch['gold_total_tags'].item()
            gold_acc += gold_total_tags
            # print("gold total tags:", gold_total_tags)
            probs = []
            for sentence in batch['sentences']:
                input_vector = sentence['input_vector']
                input_vector = input_vector.to(device)
                length = sentence['length']
                length = length.to(device)

                # TODO: unsure how this all fits together
                output = model(input_vector, length)
                output = softmax_fn(output)

                product = 1

                for idx in range(length):
                    prob = output[0, idx, input_vector.view(-1)[idx]]
                    product *= prob
                probs.append(product)
            correct_idx = torch.argmax(torch.tensor(probs))
            num_correct = int(batch['sentences'][correct_idx]['num_correct'][0])
            # print(num_correct)
            num_total = int(batch['sentences'][correct_idx]['total'][0])
            # print(num_total)

            correct_acc += num_correct
            total_acc += num_total




        precision = correct_acc / total_acc
        recall = correct_acc / gold_acc
        f1 = 2 * ( (precision*recall) / (precision + recall) )

        print("precision:", precision)
        print("recall:", recall)
        print("F1:", f1)
        experiment.log_metric("precision", precision)
        experiment.log_metric("recall", recall)
        experiment.log_metric("F1", f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("parse_file")
    parser.add_argument("gold_file")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-v", "--validate", action="store_true",
                        help="run validation loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()

    # TODO: Make sure you modify the `.comet.config` file
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    # TODO: Load dataset
    # Hint: Use random_split to split dataset into train and validate datasets

    #for training and validation
    parse_dataset = ParsingDataset(args.train_file)
    #split
    validation_size = len(parse_dataset)//10
    train_size = len(parse_dataset) - validation_size
    train_subset, validation_subset = random_split(parse_dataset, (train_size, validation_size))
    train_loader = DataLoader(train_subset, batch_size = hyperparams["batch_size"], shuffle=True)
    validate_loader = DataLoader(validation_subset, batch_size = hyperparams["batch_size"], shuffle=True)


    #for testing
    reranking_dataset = RerankingDataset(args.parse_file, args.gold_file, parse_dataset.word2id)
    test_dataset = DataLoader(reranking_dataset, batch_size = 1, shuffle=True)

    vocab_size = parse_dataset.vocab_size
    model = LSTMLM(
        vocab_size,
        hyperparams["rnn_size"],
        hyperparams["embedding_size"]
    ).to(device)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("running training loop...")
        train(model, train_loader, experiment, hyperparams)
    if args.validate:
        print("running validation...")
        validate(model, validate_loader, experiment, hyperparams)
    if args.test:
        print("testing reranker...")
        test(model, test_dataset, experiment, hyperparams)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
