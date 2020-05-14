"""First model for the sentiment analysis problem.

This file can be used as a model for all models.
"""

import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import utils

# Global variables
# ------------------------------------------

MODEL_NAME = "model_template"
MODEL_PATH = "{}{}.pt".format(utils.MODELS_FOLDER, MODEL_NAME)
device = None


class Model(nn.Module):
    """Model class.

    The model is a function taking as input a batch of strings and returning
    a scalar for each sample, representing the sentiment level.
    """

    def __init__(self):
        """Init function."""
        super(Model, self).__init__()

    def forward(self, x):
        """Forward function.

        Args:
            x (tensor): tensor of strings, shape (N,) with N the batch size
            s (tensor): tensor of scalars, shape (N,) with N the batch size
        """
        pass


def create(args):
    """Create a fresh new model.

    This function does not train or test the model. It just creates it and
    save it in models/model_name/model_name.pt.
    """
    logging.info("launching create function")

    model = Model().to(device)
    torch.save(model, MODEL_PATH)

    logging.info("model created and saved in {}".format(MODEL_PATH))


def train(args):
    """Train an existing model.

    This function trains the model models/model_name/model_name.pt. It does not
    use the testing dataset, but only the training one. The model is updated
    after each epoch.

    All operations on the training dataset are allowed.
    """
    logging.info("launching train function")

    raw_x_train = np.load(utils.X_TRAIN_PATH, allow_pickle=True)
    raw_y_train = np.load(utils.Y_TRAIN_PATH)
    logging.info("raw_x_train shape: {}".format(raw_x_train.shape))
    logging.info("raw_y_train shape: {}".format(raw_y_train.shape))

    x_train, x_valid, y_train, y_valid = train_test_split(
        raw_x_train, raw_y_train, test_size=0.2, random_state=42)
    logging.info("x_train shape: {}".format(x_train.shape))
    logging.info("y_train shape: {}".format(y_train.shape))
    logging.info("x_valid shape: {}".format(x_valid.shape))
    logging.info("y_valid shape: {}".format(y_valid.shape))

    # TODO
    # train function


def evaluate(args):
    """Evaluate an existing model.

    This function evaluates the model models/model_name/model_name.pt on the
    testing dataset. Results are saved in results/model_name/.

    The testing dataset must not be modified.
    """
    logging.info("launching evaluate function")

    x_test = np.load(utils.X_TEST_PATH, allow_pickle=True)
    y_test = np.load(utils.Y_TEST_PATH)
    logging.info("x_test shape: {}".format(x_test.shape))
    logging.info("y_test shape: {}".format(y_test.shape))

    model = torch.load(MODEL_PATH).to(device)
    model.eval()

    # TODO
    # evaluate function


if __name__ == "__main__":
    # Creating necessary folders
    # ------------------------------------------

    utils.create_dir(utils.RESULTS_FOLDER, delete_old=False)
    utils.create_dir(utils.MODELS_FOLDER, delete_old=False)

    utils.create_dir("{}{}/".format(utils.RESULTS_FOLDER, MODEL_NAME), False)
    utils.create_dir("{}{}/".format(utils.MODELS_FOLDER, MODEL_NAME), False)

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
