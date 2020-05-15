"""LSTM model for the sentiment analysis problem."""

import argparse
import logging

import numpy as np
from sklearn.model_selection import train_test_split

import utils

import pandas as pd

import time
import spacy
import random
from pathlib import Path
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data 
import torchtext
from nltk.tokenize.treebank import TreebankWordDetokenizer
from bs4 import BeautifulSoup
import re
import os

# Global variables
# ------------------------------------------

MODEL_NAME = "model_lstm"
MODEL_PATH = "{}{}.pt".format(utils.MODELS_FOLDER, MODEL_NAME)
device = None


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        """Init function.

        vocab_size - vocabulary size
        embedding_dim - size of the dense word vectors
        hidden_dim - size of the hidden states
        output_dim - number of classes
        n_layers - number of multi-layer RNN
        bidirectional - boolean - use both directions of LSTM
        dropout - dropout probability
        pad_idx -  string representing the pad token
        """
        
        super().__init__()

        # Feed in the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        # LSTM layer
        self.encoder = nn.LSTM(embedding_dim, 
                               hidden_dim, 
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               dropout=dropout)
        
        # Fully-connected layer
        self.predictor = nn.Linear(hidden_dim*2, output_dim)

        # Initialize dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, x, x_lengths):
        """Forward function.
        Args:
            x (tensor): tensor of strings, shape (N,) with N the batch size
            s (tensor): tensor of scalars, shape (N,) with N the batch size
        """
        embedded = self.dropout(self.embedding(x))    

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, x_lengths) #pack the embeddings

        packed_output, (hidden, cell) = self.encoder(packed_embedded) #encoder output

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output) #unpack sequence, transform to a tensor

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)) #final layer forward and backward hidden states

        return self.predictor(hidden)


def create(TEXT):
    """Create a new model.
    """
    logging.info("launching create function")

    #model = Model().to(device)
    #model = LSTM()
    #torch.save(model.state_dict(), MODEL_PATH)
    
    model = LSTM(vocab_size = len(TEXT.vocab), #Model LSTM
                 embedding_dim = 100, 
                 hidden_dim = 256, 
                 output_dim = 1, 
                 n_layers = 2, 
                 bidirectional = True, 
                 dropout = 0.5, 
                 pad_idx = TEXT.vocab.stoi[TEXT.pad_token])
    
    optimizer = optim.Adam(model.parameters(), lr=2e-2) #Training parameters
    criterion = nn.BCEWithLogitsLoss()

    logging.info("model created")
    
    return model, optimizer, criterion
    
def batch_accuracy(predictions, label):
    """Compute accuracy per batch.
    """
    logging.info("launching batch accuracy")

    preds = torch.round(torch.sigmoid(predictions)) # Round predictions to the closest integer using the sigmoid function
    
    correct = (preds == label).float() # If prediction is equal to label
   
    accuracy = correct.sum() / len(correct)  # Average correct predictions

    return accuracy


def batch_train(model, iterator, optimizer, criterion):
    """Train per batch
    """
    logging.info("launching train function")
    
    training_loss = 0.0 # Cumulated Training loss
    
    training_acc = 0.0 # Cumulated Training accuracy
    
    model.train() # Set model to training mode
    
    for batch in iterator: # For each batch in the training iterator
        
        optimizer.zero_grad() #zero gradients
        
        x, x_lengths = batch.text #tuple (tensor, len of seq)
        
        predictions = model(x, x_lengths).squeeze(1) #compute prediction
        
        loss = criterion(predictions, batch.label) #compute loss
        
        accuracy = batch_accuracy(predictions, batch.label) #compute accuracy
        
        loss.backward() #compute gradients
        
        optimizer.step() #optmizer
        
        training_loss += loss.item()
        training_acc += accuracy.item()
    
    return training_loss / len(iterator), training_acc / len(iterator)

def train(model, optimizer, criterion, train_iterator, valid_iterator):
    """Train an existing model.
    This function trains the model models/model_name/model_name.pt. It does not
    use the testing dataset, but only the training one. The model is updated
    after each epoch.
    """
    logging.info("launching train")
    
    best_valid_loss = float('inf') # lowest valid lost
    
    for epoch in range(1):
    
        train_loss, train_acc = batch_train(model, train_iterator, optimizer, criterion) #training loss and accuracy
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)#validation loss and accuracy
    
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_PATH)
            #create(model) #save the model if the valid lost is the best
    
        print("Epoch {}:".format(epoch+1))
        print("\t Train Loss {} | Train Accuracy: {}%".format(round(train_loss, 2), round(train_acc*100, 2)))
        print("\t Validation Loss {} | Validation Accuracy: {}%".format(round(valid_loss, 2), round(valid_acc*100, 2)))


def evaluate(model, iterator, criterion):
    """Evaluate an existing model.
    This function evaluates the model models/model_name/model_name.pt on the
    testing dataset. Results are saved in results/model_name/.
    The testing dataset must not be modified.
    """
    logging.info("launching evaluate function")
    
    eval_loss = 0.0 #training loss

    eval_acc = 0 #training accuracy

    model.eval()
    
    with torch.no_grad(): # Don't calculate the gradients
    
        for batch in iterator:

            x, x_lengths = batch.text
            
            predictions = model(x, x_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            accuracy = batch_accuracy(predictions, batch.label)

            eval_loss += loss.item()
            eval_acc += accuracy.item()
        
    return eval_loss / len(iterator), eval_acc / len(iterator)
    
def load_train():
    """Load train dataset and transform to torch TabularDataset
    """
    x_train = np.load(utils.X_TRAIN_PATH)
    y_train = np.load(utils.Y_TRAIN_PATH)
        
    index = [] #delete empty lines
    for i in range(len(x_train)):
        if x_train[i] =='':
             index.append(i)
    x_train_clean = np.delete(x_train, index) # 3247 empty cells
    y_train = np.delete(y_train, index)
    
    x_train1, x_train2, y_train1, y_train2 = train_test_split(
    x_train_clean, y_train, test_size=0.1, random_state=42)
        
    dataframe = pd.concat([pd.DataFrame(x_train2), pd.DataFrame(y_train2)], axis=1)
    pd.DataFrame(dataframe).to_csv("train_set.csv", index=None)
        
    TEXT = data.Field(tokenize='spacy', lower=True, include_lengths= True)
    LABEL = data.LabelField(dtype=torch.float)
        
    fields = [('text', TEXT),('label', LABEL)]
        
    dataset = torchtext.data.TabularDataset(
        path="train_set.csv",
        format="csv",
        fields=fields,
        skip_header=False)
    os.remove('train_set.csv')
    
    #dataset = torchtext.data.Dataset(
    #    dataframe,
    #    fields=fields)
    
    (x_train, x_valid, x_test) = dataset.split(split_ratio=[0.8,0.1,0.1])
    
    TEXT.build_vocab(x_train, 
                     max_size = 25000,
                     vectors = "glove.6B.100d",
                     unk_init = torch.Tensor.normal_)
    
    LABEL.build_vocab(x_train)
    
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (x_train, x_valid, x_test),
        batch_size = 128,
        sort_key = lambda x: len(x.text),
        sort_within_batch = True)
   
    return train_iterator, valid_iterator, test_iterator, TEXT
        
def load_test():
    """Load test dataset and transform to torch TabularDataset
    """
    x_test = np.load(utils.X_TEST_PATH)
    y_test = np.load(utils.Y_TEST_PATH)
        
    TEXT = data.Field(tokenize='spacy', lower=True, include_lengths= True)
    LABEL = data.LabelField(dtype=torch.float)
        
    fields = [('text', TEXT),('label', LABEL)]
        
    dataframe = pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test)], axis=1)
    pd.DataFrame(dataframe).to_csv("test_set.csv", index=None)
    dataset = torchtext.data.TabularDataset(
        path="test_set.csv",
        format="csv",
        fields=fields,
        skip_header=False)
    os.remove('test_set.csv')
    
    (testdata1, testdata2) = dataset.split(split_ratio=[0.9,0.1]) #test a larger and smaller set
        
    TEXT.build_vocab(testdata1, 
                     max_size = 250000,
                     vectors = "glove.6B.100d",
                     unk_init = torch.Tensor.normal_)
        
    LABEL.build_vocab(testdata1)
        
    test_iterator1, test_iterator2 = data.BucketIterator.splits(
        (testdata1, testdata2),
        batch_size = 128,
        sort_key = lambda x: len(x.text),
        sort_within_batch = True)
        
    return test_iterator1, test_iterator2

def evaluate_test(model, test_iterator2, criterion):
    """Evaluate for test dataset.
    """
    
    test_loss, test_acc = evaluate(model, test_iterator2, criterion) # Evaluate test loss and accuracy
    
    print("Test Loss: {} | Test Acc: {}%".format(round(test_loss, 2), round(test_acc*100, 2)))

if __name__ == "__main__":
    # Logging
    # ------------------------------------------
    logging.basicConfig(
        filename="{}{}/{}.log".format(utils.RESULTS_FOLDER,
                                      MODEL_NAME, MODEL_NAME),
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    # Creating necessary folders
    # ------------------------------------------

    utils.create_dir(utils.RESULTS_FOLDER, delete_old=False)
    utils.create_dir(utils.MODELS_FOLDER, delete_old=False)

    utils.create_dir("{}{}/".format(utils.RESULTS_FOLDER, MODEL_NAME), False)
    utils.create_dir("{}{}/".format(utils.MODELS_FOLDER, MODEL_NAME), False)

    # Main parser
    # ------------------------------------------

    parser = argparse.ArgumentParser(
        "{} for the sentiment analysis problem".format(MODEL_NAME))
    parser.add_argument("--cuda", help="Using GPU or not", action="store_true")
    subparsers = parser.add_subparsers()

    train_iterator, valid_iterator, test_iterator, TEXT = load_train()
    
    # Create parser
    # ------------------------------------------

    parser_create = subparsers.add_parser(
        "create",
        help="Command to create a fresh new model",
    )
    parser_create.set_defaults(func=create)
    
    model, optimizer, criterion = create(TEXT)
    
    # Train parser
    # ------------------------------------------

    parser_train = subparsers.add_parser(
        "train",
        help="Command to train the model",
    )
    parser_train.add_argument(
        "--epochs", help="Number of epochs", default=42, type=int)
    parser_train.add_argument(
        "--batch-size", help="Batch size", default=128, type=int)
    parser_train.set_defaults(func=train)
    
    train(model, optimizer, criterion, train_iterator, valid_iterator)

    # Evaluate parser
    # ------------------------------------------

    parser_evaluate = subparsers.add_parser(
        "evaluate",
        help="Command to evaluate an existing model"
    )
    parser_evaluate.set_defaults(func=evaluate_test)
    
    #model.load_state_dict(torch.load('model-small.pt')) # Load the model with the best validation loss

    test_iterator1, test_iterator2 = load_test()    
    
    evaluate_test(model, test_iterator2, criterion)

    # Parse arguments
    # ------------------------------------------

    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda_is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #args.func(args)