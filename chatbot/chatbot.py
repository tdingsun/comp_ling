from comet_ml import Experiment
import torch
import torch.nn as nn
import argparse
import math
import numpy as np
from preprocess import *
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_params = {
     "batch_size": 25,
     "num_epochs": 2,
     "learning_rate": 0.000001,
     "window_size": 100
 }


def train(model, train_loader, optimizer, experiment, pad_index):
    """
    Trains the model.
    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param optimizer: the initilized optimizer
    :param experiment: comet.ml experiment object
    """
    # TODO: Write the training loop here, save trained model weights if needed
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
    model = model.train()

    with experiment.train():
        for epoch in range(hyper_params['num_epochs']):
            total_loss = 0
            word_count = 0
            batch_num = 0
            for batch in tqdm(train_loader):
                x = batch['input_vectors'].to(DEVICE)
                y = batch['label_vectors'].to(DEVICE)
                outputs = model(x, labels=x)
                _, logits = outputs[:2]
                myLoss = loss_fn(torch.flatten(logits, 0, 1), torch.flatten(y, 0, 1))
                myLoss.backward()
                optimizer.step()
                print(myLoss)

                lengths = batch['lengths']
                num_words_in_batch = torch.sum(lengths).item()
                batch_loss = myLoss.item()*num_words_in_batch

                total_loss += batch_loss
                word_count += num_words_in_batch
        
            perplexity = np.exp(total_loss / word_count)
            print("perplexity:", perplexity)
            experiment.log_metric("perplexity", perplexity)
            torch.save(model.state_dict(), 'model.pt')




def test(model, test_loader, experiment, pad_index):
    """
    Validates the model performance as LM on never-seen data using perplexity.
    :param model: the trained model to use for testing
    :param test_loader: Dataloader of testing data
    :param experiment: comet.ml experiment object
    """
    total_loss = 0
    word_count = 0
    # TODO: Write the testing loop and calculate perplexity
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
    model = model.eval()
    with experiment.validate():
        for batch in tqdm(test_loader):
            x = batch['input_vectors'].to(DEVICE)
            y = batch['label_vectors'].to(DEVICE)
            outputs = model(x, labels=x)
            _, logits = outputs[:2]
            myLoss = loss_fn(torch.flatten(logits, 0 , 1), torch.flatten(y, 0, 1))
            print(myLoss)

            lengths = batch['lengths']
            num_words_in_batch = torch.sum(lengths).item()
            total_loss += myLoss.item()*num_words_in_batch
            word_count += num_words_in_batch
        
        perplexity = np.exp(total_loss / word_count)
        print("perplexity:", perplexity)
        experiment.log_metric("perplexity", perplexity)


def interactive(input, tokenizer, model, top_k=10, ntok=20):
    """
    Generate and print out the response given input using the trained model
    :param input: an input string as prompt (i.e. How are you?)
    :param tokenizer: intialized tokenizer object for encoding the input
    :param model: the trained model to use for generate prediction
    :param top_k: number of samples for top_k sampling
    :param ntok: maximum number of tokens to generate

    Comment: Feed in the input to the model to generate the most probable token
    and concatenate it with current input.
    Continue this process iteratively until the model predicts the padding
    token or reach the maximum number of tokens.
    You may need to add the BOS token and special token to the input sentence
    before passing into model.
    Also, you may want to filter out your input sentence and meaningless tokens
    when printing out the response.
    """
    # TODO: Write the generation function for interacting with trained model
    encoded_input = torch.tensor(tokenizer.encode(tokenizer.bos_token + input + " " + tokenizer.sep_token)).to(DEVICE)
    print(encoded_input)
    for i in range(ntok):
        outputs = model(encoded_input)
        predictions = outputs[0]
        print(predictions.shape)
        topk = torch.topk(predictions[-1, :], top_k)
        encoded_input += [topk.indices[0]]

    response = tokenizer.decode(encoded_input)
    print(response)


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
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="run in interactive mode")
    args = parser.parse_args()

    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyper_params)

    # Load the GPT2 Tokenizer, add any special token if needed
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens_dict = {'sep_token': '<SEP>', 'pad_token': '<PAD>'}
    tokenizer.add_special_tokens(special_tokens_dict)
    pad_index = tokenizer.pad_token_id

    # Intialized the pretrained GPT-2 model and optimizer
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["learning_rate"])
    # Load the train, test DataLoader NOTE: Parse the data using GPT2 tokenizer

    if not args.interactive:
        train_loader, test_loader = load_dataset(args.train_file, args.test_file, tokenizer, hyper_params["batch_size"], hyper_params["window_size"])
    
    if args.load:
        model.load_state_dict(torch.load('model.pt'))
        print("model loaded")
    if args.train:
        # run train loop here
        print("running training loop...")
        train(model, train_loader, optimizer, experiment, pad_index)
    if args.save:
        torch.save(model.state_dict(), 'model.pt')
    if args.test:
        # run test loop here
        print("running testing loop...")
        test(model, test_loader, experiment, pad_index)
    if args.interactive:
        # generate your own chat with the model here
        print("running interctive mode...")
        while True:
            input_text = input("Please say something: ")
            interactive(input_text, tokenizer, model)
