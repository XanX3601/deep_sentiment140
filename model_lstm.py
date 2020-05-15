"""LSTM model for the sentiment analysis problem."""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from sklearn.model_selection import train_test_split
from torchtext import data

import utils

# Global variables
# ------------------------------------------

MODEL_NAME = "model_lstm"
MODEL_PATH = "{}{}.pt".format(utils.MODELS_FOLDER, MODEL_NAME)
CRITERION = nn.BCEWithLogitsLoss()
device = None


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        pad_idx
    ):
        """Init function.

        Args:
            vocab_size (int): vocabulary size.
            embedding_dim (int): size of the dense word vectors.
            hidden_dim (int): size of the hidden states.
            output_dim (int): number of classes.
            n_layers (int): number of multi-layer RNN.
            bidirectional (bool): use both directions of LSTM.
            dropout (float): dropout probability.
            pad_idx (str): string representing the pad token.
        """

        super().__init__()

        # Feed in the embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

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
            x (tensor): tensor of strings, shape (N,) with N the batch size.
            x_lengths (tensor): tensor of scalars, shape (N,) with N the batch
                size.
        """
        embedded = self.dropout(self.embedding(x))

        # Pack the embeddings
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, x_lengths)

        # Encoder output
        packed_output, (hidden, cell) = self.encoder(packed_embedded)

        # Unpack sequence, transform to a tensor
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output)

        # Final layer forward and backward hidden states
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.predictor(hidden)


def load_train():
    """Load train dataset and transform to torch Dataset.

    Returns:
        train_iterator (IterableDataset): for train.
        valid_iterator (IterableDataset): for valid.
        test_iterator (IterableDataset): for test.
        TEXT (torchtext.data.field): text dataset.
    """
    x_train = np.load(utils.X_TRAIN_PATH)
    y_train = np.load(utils.Y_TRAIN_PATH)

    x_train1, x_train2, y_train1, y_train2 = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42)

    dataframe = pd.concat(
        [pd.DataFrame(x_train2), pd.DataFrame(y_train2)], axis=1)
    pd.DataFrame(dataframe).to_csv("train_set.csv", index=None)

    TEXT = data.Field(tokenize="spacy", lower=True, include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float)

    fields = [("text", TEXT), ("label", LABEL)]

    dataset = torchtext.data.TabularDataset(
        path="train_set.csv",
        format="csv",
        fields=fields,
        skip_header=False)
    os.remove("train_set.csv")

    (x_train, x_valid, x_test) = dataset.split(split_ratio=[0.8, 0.1, 0.1])

    TEXT.build_vocab(x_train,
                     max_size=25000,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(x_train)

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (x_train, x_valid, x_test),
        batch_size=128,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True)

    return train_iterator, valid_iterator, test_iterator, TEXT


def load_test():
    """Load test dataset and transform to torch Dataset.

    Returns:
        test_iterator1 (IterableDataset): big test set.
        test_iterator2 (IterableDataset): small test set.
    """
    x_test = np.load(utils.X_TEST_PATH)
    y_test = np.load(utils.Y_TEST_PATH)

    TEXT = data.Field(tokenize="spacy", lower=True, include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float)

    fields = [("text", TEXT), ("label", LABEL)]

    dataframe = pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test)], axis=1)
    pd.DataFrame(dataframe).to_csv("test_set.csv", index=None)
    dataset = torchtext.data.TabularDataset(
        path="test_set.csv",
        format="csv",
        fields=fields,
        skip_header=False)
    os.remove("test_set.csv")

    # Test a larger and smaller set
    (testdata1, testdata2) = dataset.split(
        split_ratio=[0.9, 0.1])

    TEXT.build_vocab(testdata1,
                     max_size=250000,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(testdata1)

    test_iterator1, test_iterator2 = data.BucketIterator.splits(
        (testdata1, testdata2),
        batch_size=128,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True)

    return test_iterator1, test_iterator2


def create(args):
    """Create a fresh new model.

    This function does not train or test the model. It just creates it and
    save it in models/model_name/model_name.pt.
    """
    logging.info("launching create function")

    train_iterator, valid_iterator, test_iterator, TEXT = load_train()

    model = LSTM(vocab_size=len(TEXT.vocab),
                 embedding_dim=100,
                 hidden_dim=256,
                 output_dim=1,
                 n_layers=2,
                 bidirectional=True,
                 dropout=0.5,
                 pad_idx=TEXT.vocab.stoi[TEXT.pad_token]).to(device)

    torch.save(model, MODEL_PATH)
    logging.info("model created and saved in {}".format(MODEL_PATH))


def batch_accuracy(predictions, label):
    """Compute accuracy per batch."""
    # Round predictions to the closest integer using the sigmoid function
    preds = torch.round(torch.sigmoid(predictions))

    # If prediction is equal to label
    correct = (preds == label).float()

    # Average correct predictions
    accuracy = correct.sum() / len(correct)

    return accuracy


def batch_train(model, iterator, optimizer, criterion):
    """Train per batch."""
    # Cumulated Training loss
    training_loss = 0.0

    # Cumulated Training accuracy
    training_acc = 0.0

    # Set model to training mode
    model.train()

    # For each batch in the training iterator
    for batch in iterator:

        optimizer.zero_grad()

        # tuple (tensor, len of seq)
        x, x_lengths = batch.text

        predictions = model(x, x_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        accuracy = batch_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        training_acc += accuracy.item()

    return training_loss / len(iterator), training_acc / len(iterator)


def train(args):
    """Train an existing model.

    This function trains the model models/model_name/model_name.pt. It does not
    use the testing dataset, but only the training one. The model is updated
    after each epoch.
    """
    logging.info("launching train")

    model = torch.load(MODEL_PATH).to(device)
    train_iterator, valid_iterator, test_iterator, TEXT = load_train()

    # Lowest valid lost
    best_valid_loss = float("inf")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=2e-2)

    for epoch in range(args.epochs):

        # Training loss and accuracy
        train_loss, train_acc = batch_train(
            model, train_iterator, optimizer, CRITERION)

        # Validation loss and accuracy
        valid_loss, valid_acc = evaluate_(model, valid_iterator)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, MODEL_PATH)

        print("Epoch {}:".format(epoch+1))
        print("\t Train Loss {} | Train Accuracy: {}%".format(
            round(train_loss, 2), round(train_acc*100, 2)))
        print("\t Validation Loss {} | Validation Accuracy: {}%".format(
            round(valid_loss, 2), round(valid_acc*100, 2)))


def evaluate_(model, iterator):
    """Evaluate an existing model.

    This function evaluates the model models/model_name/model_name.pt on the
    testing dataset. Results are saved in results/model_name/.

    The testing dataset must not be modified.
    """
    logging.info("launching evaluate function")

    eval_loss = 0.0
    eval_acc = 0
    model.eval()

    # Do not calculate the gradients
    with torch.no_grad():

        for batch in iterator:

            x, x_lengths = batch.text
            predictions = model(x, x_lengths).squeeze(1)
            loss = CRITERION(predictions, batch.label)
            accuracy = batch_accuracy(predictions, batch.label)

            eval_loss += loss.item()
            eval_acc += accuracy.item()

    return eval_loss / len(iterator), eval_acc / len(iterator)


def evaluate(args):
    """Evaluate an existing model.

    This function evaluates the model models/model_name/model_name.pt on the
    testing dataset. Results are saved in results/model_name/.

    The testing dataset must not be modified.
    """
    model = torch.load(MODEL_PATH).to(device)
    test_iterator1, test_iterator2 = load_test()

    # Evaluate test loss and accuracy
    test_loss, test_acc = evaluate_(model, test_iterator2)

    print("Test Loss: {} | Test Acc: {}%".format(
        round(test_loss, 2), round(test_acc*100, 2)))


if __name__ == "__main__":
    # Creating necessary folders
    # ------------------------------------------

    utils.create_dir(utils.RESULTS_FOLDER, delete_old=False)
    utils.create_dir(utils.MODELS_FOLDER, delete_old=False)

    utils.create_dir("{}{}/".format(utils.RESULTS_FOLDER, MODEL_NAME), False)

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

    # Main parser
    # ------------------------------------------

    parser = argparse.ArgumentParser(
        "{} for the sentiment analysis problem".format(MODEL_NAME))
    parser.add_argument("--cuda", help="Using GPU or not", action="store_true")
    subparsers = parser.add_subparsers()

    # Create parser
    # ------------------------------------------

    parser_create = subparsers.add_parser(
        "create",
        help="Command to create a fresh new model",
    )
    parser_create.set_defaults(func=create)

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

    # Evaluate parser
    # ------------------------------------------

    parser_evaluate = subparsers.add_parser(
        "evaluate",
        help="Command to evaluate an existing model"
    )
    parser_evaluate.set_defaults(func=evaluate)

    # Parse arguments
    # ------------------------------------------

    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda_is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    args.func(args)
