from comet_ml import Experiment
from preprocess import MyDataset, read_file, create_dicts, tokenize
from model import CharLM
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm  # optional progress bar

# TODO: Set hyperparameters
hyperparams = {
    "num_epochs": 1,
    "lr": 1.0,
    "lstm_batch_size": 20,
    "lstm_seq_len": 35,
    "word_embed_size": 650,
    "char_embed_size": 15
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(myModel, train_loader, loss_fn, word2id, experiment, hyperparams):
    """
    Training loop that trains BERT model.

    Inputs:
    - model: A BERT model
    - train_loader: Dataloader of training data
    - experiment: comet.ml experiment object
    - hyperparams: Hyperparameters dictionary
    """

    learning_rate = hyperparams["lr"]
    old_perplexity = 100000
    best_perplexity = 100000

    hidden = (Variable(torch.zeros(2, hyperparams['lstm_batch_size'], hyperparams['word_embed_size'])).to(device), 
              Variable(torch.zeros(2, hyperparams['lstm_batch_size'], hyperparams['word_embed_size'])).to(device))

    with experiment.train():
        for epoch in range(hyperparams["num_epochs"]):
            ##### VALIDATION #####
            myModel = myModel.eval()
            loss_batch = []
            perplexity_batch = []

            for batch in tqdm(valid_loader):
                x = batch['input_vecs'].to(device)
                y = batch['label_vecs'].to(device)
                hidden = [state.detach() for state in hidden]
                v_output, hidden = myModel(x, hidden)
                y = y.contiguous().view(-1)

                loss = loss_fn(v_output, y)
                perplexity = torch.exp(loss.data)
                loss_batch.append(float(loss))
                perplexity_batch.append(float(perplexity))
            
            perplexity = np.mean(perplexity_batch)
            loss = np.mean(loss_batch)
            print("[epoch {}] validation perplexity={}".format(epoch, perplexity))
            print("validation loss={}".format(loss))
            print("perplexity decrease={}".format(float(old_perplexity - perplexity)))
            experiment.log_metric("perplexity", perplexity)
            experiment.log_metric("loss", loss)

            if best_perplexity > perplexity:
                best_perplexity = perplexity
                # print("saving model")
                # torch.save(myModel.state_dict(), './saved_model.pt')

            if float(old_perplexity - perplexity) <= 1.0:
                learning_rate /= 2
                print("halved lr:{}".format(learning_rate))
            
            old_perplexity = perplexity

            ##### TRAINING #####
            myModel = myModel.train()
            optimizer = torch.optim.SGD(myModel.parameters(), lr = learning_rate, momentum = 0.85)
            for batch in tqdm(train_loader):
                optimizer.zero_grad()

                x = batch['input_vecs'].to(device)
                y = batch['label_vecs'].to(device)
                hidden = [state.detach() for state in hidden]
                output, hidden = myModel(x, hidden)
                y = y.contiguous().view(-1)

                loss = loss_fn(output, y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(myModel.parameters(), 5, norm_type=2)
                optimizer.step()

        # print("saving model")
        # torch.save(myModel.state_dict(), './saved_model.pt')
        print("Training finished")


def test(myModel, test_loader, loss_fn, word2id, experiment, hyperparams):
    """
    Testing loop for BERT model and logs perplexity and accuracy to comet.ml.

    Inputs:
    - model: A BERT model
    - train_loader: Dataloader of training data
    - experiment: comet.ml experiment object
    - hyperparams: Hyperparameters dictionary
    """
    myModel = myModel.eval()
    loss_batch = []
    perplexity_batch = []

    hidden = (Variable(torch.zeros(2, hyperparams['lstm_batch_size'], hyperparams['word_embed_size'])).to(device), 
              Variable(torch.zeros(2, hyperparams['lstm_batch_size'], hyperparams['word_embed_size'])).to(device))

    with experiment.test(), torch.no_grad():

        for batch in tqdm(test_loader):
            x = batch['input_vecs'].to(device)
            y = batch['label_vecs'].to(device)
            hidden = [state.detach() for state in hidden]

            test_output, hidden = myModel(x, hidden)
            y = y.contiguous().view(-1)

            loss = loss_fn(test_output, y)
            perplexity = torch.exp(loss.data)
            loss_batch.append(float(loss))
            perplexity_batch.append(float(perplexity))
        
        perplexity = np.mean(perplexity_batch)
        loss = np.mean(loss_batch)
        print("test perplexity: ", perplexity)
        print("test loss: ", loss)
        
        experiment.log_metric("perplexity", perplexity)

def generate(input_text, myModel, experiment, char2id, max_word_len, word2id, id2word, device, ntok=100, top_k=1):
    hidden = (Variable(torch.zeros(2, 1, hyperparams['word_embed_size'])).to(device), 
              Variable(torch.zeros(2, 1, hyperparams['word_embed_size'])).to(device))

    input_text = "*STOP* " + input_text
    input_seq = tokenize(input_text.split(), char2id, max_word_len)
    output_seq = []
    for i in range(ntok):
        x = torch.tensor(input_seq).to(device)
        x = x.view(1, -1, max_word_len+2)
        hidden = [state.detach() for state in hidden]
        output, hidden = myModel(x, hidden, generate=True)
        topk = torch.topk(output[-1, :], top_k).indices
        rand = random.randint(0, top_k-1)
        chosen_index = topk[rand].item()
        input_seq += tokenize([id2word[chosen_index]], char2id, max_word_len)
        output_seq += [chosen_index]
    
    decoded_output = [id2word[word] for word in output_seq]
    print(input_text + " " + " ".join(decoded_output))

def crawl(input_text, myModel, experiment, char2id, max_word_len, word2id, id2word, device, ntok=500, top_k=10):
    input_text = "*STOP* " + input_text
    input_seq = tokenize(input_text.split(), char2id, max_word_len)
    output_seq = []
    x = torch.tensor(input_seq).to(device)
    x = x.view(1, -1, max_word_len+2)
    embedding = myModel.getEmbedding(x)
    print(embedding.shape)
    for i in range(ntok):
        embedding += torch.randn(embedding.size()[0], embedding.size()[1]).to(device) * 2048.0
        logits = myModel.getWordFromEmbedding(embedding)
        topk = torch.topk(logits[-1, :], top_k).indices
        # rand = random.randint(0, top_k-1)
        rand = 0
        chosen_index = topk[rand].item()
        input_seq += tokenize([id2word[chosen_index]], char2id, max_word_len)
        output_seq += [chosen_index]
    
    decoded_output = [id2word[word] for word in output_seq]
    print(input_text + " " + " ".join(decoded_output))

def passback(input_text, myModel, experiment, char2id, max_word_len, word2id, id2word, device, top_k=5):
    hidden = (Variable(torch.zeros(2, 1, hyperparams['word_embed_size'])).to(device), 
              Variable(torch.zeros(2, 1, hyperparams['word_embed_size'])).to(device))

    input_text = "*STOP* " + input_text
    input_seq = tokenize(input_text.split(), char2id, max_word_len)
    output_seq = input_text
    while True:
        x = torch.tensor(input_seq).to(device)
        x = x.view(1, -1, max_word_len+2)
        hidden = [state.detach() for state in hidden]
        output, hidden = myModel(x, hidden, generate=True)
        topk = torch.topk(output[-1, :], top_k).indices
        rand = random.randint(0, top_k-1)
        chosen_index = topk[rand].item()
        input_seq += tokenize([id2word[chosen_index]], char2id, max_word_len)
        output_seq += " " + id2word[chosen_index]
        print(output_seq)
        next_input = input("Next Word: ")
        input_seq += tokenize(next_input.split(), char2id, max_word_len)
        output_seq += " " + next_input
        print(output_seq)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("valid_file")
    parser.add_argument("test_file")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-g", "--generate", action="store_true",
                        help="generate words")
    parser.add_argument("-p", "--passback", action="store_true",
                        help="special function")
    parser.add_argument("-c", "--crawl", action="store_true",
                        help="special function")       
    parser.add_argument("-num_epochs", "--num_epochs", type=int)              
    args = parser.parse_args()

    print(args.num_epochs)
    hyperparams["num_epochs"] = args.num_epochs

    # Comet.ml setup
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    # Load dataset
    word2id, char2id = create_dicts(args.train_file, args.valid_file, args.test_file)

    id2word = {value:key for key, value in word2id.items()}
    max_word_len = max([len(word) for word in word2id])
    print("Max word length: ", max_word_len)
    vocab_size = len(word2id)
    print("Vocab size: ", vocab_size)
    char_vocab_size = len(char2id)
    print("Char vocab size: ", char_vocab_size)

    train_set = MyDataset(args.train_file, hyperparams['lstm_seq_len'], hyperparams['lstm_batch_size'], word2id, char2id, max_word_len)
    valid_set = MyDataset(args.valid_file, hyperparams['lstm_seq_len'], hyperparams['lstm_batch_size'], word2id, char2id, max_word_len)

    train_loader = DataLoader(train_set, batch_size=hyperparams['lstm_batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=hyperparams['lstm_batch_size'], shuffle=True)

    test_set = MyDataset(args.test_file, hyperparams['lstm_seq_len'], hyperparams['lstm_batch_size'], word2id, char2id, max_word_len)
    test_loader = DataLoader(test_set, batch_size=hyperparams['lstm_batch_size'], shuffle=True)

    # Make model
    myModel = CharLM(hyperparams["char_embed_size"], 
                    hyperparams["word_embed_size"], 
                    vocab_size, 
                    char_vocab_size,
                    hyperparams["lstm_seq_len"],
                    hyperparams["lstm_batch_size"]).to(device)
    print("Model made")

    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index = 0)

    if args.load:
        print("Loading Model")
        myModel.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("training")
        train(myModel, train_loader, loss_fn, word2id, experiment, hyperparams)
    if args.test:
        print("testing")
        test(myModel, test_loader, loss_fn, word2id, experiment, hyperparams)
    if args.save:
        print("Saving Model")
        torch.save(myModel.state_dict(), './model.pt')
    if args.generate:
        while True:
            input_text = input("Input: ")
            generate(input_text, myModel, experiment, char2id, max_word_len, word2id, id2word, device)
    if args.crawl:
        while True:
            input_text = input("Input: ")
            crawl(input_text, myModel, experiment, char2id, max_word_len, word2id, id2word, device)
    if args.passback:
        input_text = input("Input: ")
        passback(input_text, myModel, experiment, char2id, max_word_len, word2id, id2word, device)
 

 
