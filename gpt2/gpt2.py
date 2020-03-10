from comet_ml import Experiment
import torch
import torch.nn
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformer import *
from preprocess import load_transformer_dataset, load_gpt2_dataset
from tqdm import tqdm

hyper_params = {
    "batch_size": 10,
    "num_epochs": 1,
    "learning_rate": 0.001,
    "embedding_size": 256
 }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

experiment = Experiment(project_name="gpt2")
experiment.log_parameters(hyper_params)

def make_mask(window_size):
    attn_shape = (1, window_size, window_size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# Train the Model
def train_transformer(model, train_loader, experiment, hyperparams, tokenizer):
    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=(model.vocab_size - 1))
    print(model.vocab_size - 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["learning_rate"])
    model = model.train()
    with experiment.train():
        for e in range(hyper_params['num_epochs']):
            for batch in tqdm(train_loader):
                mask = make_mask(batch['input_vectors'].size(-1)).to(device)
                x = batch['input_vectors'].to(device)
                y = batch['label_vectors'].to(device)
                lengths = batch['lengths'].to(device)
                optimizer.zero_grad()
                y_pred = model(x, mask)

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


# Test the Model
def test_transformer(model, train_loader, experiment, hyperparams):
    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=(model.vocab_size - 1))
    total_loss = 0
    word_count = 0
    total_wrong = 0

    model = model.eval()
    with experiment.test():
        for batch in tqdm(test_loader):
            mask = make_mask(batch['input_vectors'].size(-1)).to(device)
            x = batch['input_vectors'].to(device)
            y = batch['label_vectors'].to(device)
            lengths = batch['lengths'].to(device)

            y_pred = model(x, mask)
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

def test_gpt2(model, test_loader, experiment, hyperparams):
    total_loss = 0
    word_count = 0
    model = model.eval()
    with experiment.test():
        for batch in tqdm(test_loader):
            x = batch['input_vectors'].to(device)
            outputs = model(x, labels=x)
            loss, logits = outputs[:2]
            num_words_in_batch = x.shape[1]
            total_loss += loss.item()*num_words_in_batch
            word_count += num_words_in_batch

        perplexity = np.exp(total_loss / word_count)
        print("perplexity: ", perplexity)
        experiment.log_metric("perplexity", perplexity)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    parser.add_argument("-m", "--model", type=str, default="",
                        help="transformer or gpt2")
    parser.add_argument("-bs", "--batch_size", type=int, default=1,
                        help="batch size")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()

    # Load the GPT2 Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    tokens_dict_transformer = {'pad_token': '<PAD>'}
    
    # Load the train, test DataLoader NOTE: Parse the data using the GPT2 tokenizer for both models

    if args.model == "transformer":
        # Load your transformer
        print("preprocessing...")
        tokenizer.add_special_tokens(tokens_dict_transformer)
        train_loader, test_loader, vocab_size, window_size = load_transformer_dataset(args.train_file, args.test_file, tokenizer, hyper_params['batch_size'])

        model = Transformer(
            vocab_size,
            window_size,
            hyper_params['batch_size'],
            hyper_params['embedding_size']
        ).to(device)
        print("preprocessing done")
        if args.train:
            print("training transformer")
            train_transformer(model, train_loader, experiment, hyper_params, tokenizer)
        if args.test:
            print("testing transformer")
            test_transformer(model, test_loader, experiment, hyper_params)
    elif args.model == "gpt2":
        # Load the GPT2 model
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        test_loader = load_gpt2_dataset(args.test_file, tokenizer, 1)
        test_gpt2(model, test_loader, experiment, hyper_params)

    # Train the model if args.model == "transformer"


    # Test the model on the test set - report perplexity
    # NOTE: For the gpt2 model, you need to feed in one token at a time.
