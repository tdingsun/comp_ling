from comet_ml import Experiment
from preprocess import MyDataset, read_file, create_dicts, tokenize
from model import CharLM
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar

# TODO: Set hyperparameters
hyperparams = {
    "num_epochs": 10,
    "lr": 1.0,
    "lstm_batch_size": 20,
    "lstm_seq_len": 35,
    "word_embed_size": 300,
    "char_embed_size": 15
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, loss_fn, word2vec, experiment, hyperparams):
    """
    Training loop that trains BERT model.

    Inputs:
    - model: A BERT model
    - train_loader: Dataloader of training data
    - experiment: comet.ml experiment object
    - hyperparams: Hyperparameters dictionary
    """

    learning_rate = hyperparams[lr]
    old_ppl = 100000
    best_ppl = 100000

    hidden = (to_var(torch.zeros(2, hyperparams['lstm_batch_size'], hyperparams['word_embed_size'])), 
              to_var(torch.zeros(2, hyperparams['lstm_batch_size'], hyperparams['word_embed_size'])))

    with experiment.train():
        batch_id = 0
        
        for epoch in range(hyperparams["num_epochs"]):
            # VALIDATION
            model = model.eval()
            loss_batch = []
            ppl_batch = []

            for batch in tqdm(valid_loader):
                x = batch['input_vecs'].to(device)
                y = batch['label_vecs'].to(device)
                hidden = [state.detach() for state in hidden]
                v_output, hidden = model(x, hidden)

                loss = loss_fn(torch.flatten(v_output, 0, 1), torch.flatten(y, 0, 1))
                ppl = torch.exp(loss.data)
                loss_batch.append(float(loss))
                ppl_batch.append(float(ppl))
            
            ppl = np.mean(ppl_batch)
            print("[epoch {}] valid PPL={}".format(epoch, PPL))
            print("valid loss={}".format(np.mean(loss_batch)))
            print("PPL decrease={}".format(float(old_PPL - PPL)))

            if best_ppl > ppl:
                best_ppl = ppl
                torch.save(model.state_dict(), './model.pt')

            if float(old_ppl - ppl) <= 1.0:
                learning_rate /= 2
                print("halved lr:{}".format(learning_rate))
            
            old_ppl = ppl

            # TRAINING
            model = model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.85)
            for batch in tqdm(train_loader):
                optimizer.zero_grad()

                x = batch['input_vecs'].to(device)
                y = batch['label_vecs'].to(device)
                mask = y != 0 #true where not zero, flase everywhere else
                hidden = [state.detach() for state in hidden]
                output, hidden = model(x, hidden)

                loss = loss_fn(torch.flatten(y_pred, 0, 1), torch.flatten(y, 0, 1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 5, norm_type=2)
                optimizer.step()

                if (batch_id + 1) % 100 == 0:
                    print(loss.item())
                batch_id += 1
        torch.save(model.state_dict(), './model.pt')
        print("Training finished")


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

    hidden = (to_var(torch.zeros(2, hyperparams['lstm_batch_size'], hyperparams['word_embed_size'])), 
              to_var(torch.zeros(2, hyperparams['lstm_batch_size'], hyperparams['word_embed_size'])))

    with experiment.test(), torch.no_grad():

        for batch in tqdm(test_loader):
            x = batch['input_vecs'].to(device)
            y = batch['label_vecs'].to(device)
            hidden = [state.detach() for state in hidden]

            test_output, hidden = model(x, hidden)
            loss = loss_fn(torch.flatten(test_output, 0, 1), torch.flatten(y, 0, 1)).data
            total_loss += loss
            word_count += 1
            
        perplexity = np.exp(total_loss / word_count)
        print(perplexity)
        experiment.log_metric("perplexity", perplexity)

def generate(model, experiment, train_set, test_set, batch_size, word2id, device):
    pass


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
                        help="run embedding analysis")
    args = parser.parse_args()

    # TODO: Make sure you modify the `.comet.config` file
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    # Load dataset
    word2id, char2id = create_dicts(args.train_file, args.valid_file, args.test_file)
    id2word = {value:key for key, value in word2id.items()}
    max_word_len = max([len(word) for word in word2id])
    print(max_word_len)
    vocab_size = len(word2id)
    print(vocab_size)
    char_vocab_size = len(char2id)
    print(char_vocab_size)

    char2id["*BOW*"] = len(char2id) + 1
    char2id["*EOW*"] = len(char2id) + 1
    char2id["*PAD*"] = 0

    train_set = MyDataset(args.train_file, hyperparams['lstm_seq_len'], word2id, char2id, max_word_len)
    valid_set = MyDataset(args.valid_file, hyperparams['lstm_seq_len'], word2id, char2id, max_word_len)
    test_set = MyDataset(args.test_file, hyperparams['lstm_seq_len'], word2id, char2id, max_word_len)

    train_loader = DataLoader(train_set, batch_size=hyperparams['lstm_batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=hyperparams['lstm_batch_size'], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=hyperparams['lstm_batch_size'], shuffle=False)

    # Make model
    model = CharLM(hyperparams["char_embed_size"], 
                    hyperparams["word_embed_size"], 
                    vocab_size, 
                    char_vocab_size,
                    hyperparams["lstm_seq_len"],
                    hyperparams["lstm_batch_size"]).to(device)
    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index = 0)

    if args.load:
        print("loading model")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        train(model, train_loader, loss_fn, word2vec, experiment, hyperparams)
    if args.test:
        test(model, test_loader, loss_fn, word2vec, experiment, hyperparams)
    if args.save:
        torch.save(model.state_dict(), './model.pt')
    if args.generate:
        generate(model, experiment, train_set, test_set,
                           hyperparams["batch_size"], word2id, device)