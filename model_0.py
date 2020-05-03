"""First model for the sentiment analysis problem."""

import argparse
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.modules as module
from sklearn.model_selection import train_test_split

import utils


def p(text):
    """Text preprocessing for logging.

    Args:
        text (str): text log

    Returns:
        p_text (str): preprocessed text
    """
    p_text = "{} - {}".format(
        datetime.today().strftime("%Y-%m-%d-%H:%M:%S"), text)

    return p_text


def create(args):
    """Function to create a fresh new model_0.

    This function does not train or test the model. It just creates it and
    save it in models/model_0/models_0.pt.
    """
    logging.INFO(p("launching create function"))


def train(args):
    """Function to train an existing model_0.

    This function trains the model models/model_0/models_0.pt. It does not use
    the testing dataset, but only the training one. The model is updated
    after each epoch.

    All operations on the training dataset are allowed.
    """
    logging.INFO(p("launching train function"))

    raw_x_train = np.load(utils.X_TRAIN_PATH, allow_pickle=True)
    raw_y_train = np.load(utils.Y_TRAIN_PATH)
    logging.INFO(p("raw_x_train shape: {}".format(raw_x_train.shape)))
    logging.INFO(p("raw_y_train shape: {}".format(raw_y_train.shape)))

    x_train, x_valid, y_train, y_valid = train_test_split(
        raw_x_train, raw_y_train, test_size=0.2, random_state=42)
    logging.INFO(p("x_train shape: {}".format(x_train.shape)))
    logging.INFO(p("y_train shape: {}".format(y_train.shape)))
    logging.INFO(p("x_valid shape: {}".format(x_valid.shape)))
    logging.INFO(p("y_valid shape: {}".format(y_valid.shape)))


def evaluate(args):
    """Function to evaluate an existing model_0.

    This function evaluates the model models/model_0/models_0.pt on the testing
    dataset. Results are saved in results/model_0/.

    The testing dataset must not be modified.
    """
    logging.INFO(p("launching evaluate function"))

    x_test = np.load(utils.X_TEST_PATH, allow_pickle=True)
    y_test = np.load(utils.Y_TEST_PATH)
    logging.INFO(p("x_test shape: {}".format(x_test.shape)))
    logging.INFO(p("y_test shape: {}".format(y_test.shape)))


if __name__ == "__main__":
    # Logging
    # ------------------------------------------
    logging.basicConfig(
        filename="{}model_0/model_0.log".format(utils.RESULTS_FOLDER),
        filemode="a",
        level=logging.DEBUG
    )

    # Creating necessary folders
    # ------------------------------------------

    utils.create_dir(utils.RESULTS_FOLDER, delete_old=False)
    utils.create_dir(utils.MODELS_FOLDER, delete_old=False)

    utils.create_dir("{}model_0/".format(utils.RESULTS_FOLDER), False)
    utils.create_dir("{}model_0/".format(utils.MODELS_FOLDER), False)

    # Main parser
    # ------------------------------------------

    parser = argparse.ArgumentParser(
        "model_0 for the sentiment analysis problem")
    subparsers = parser.add_subparsers()

    # Create parser
    # ------------------------------------------

    parser_create = subparsers.add_parser(
        "create",
        help="Command to create a fresh new model_0",
    )
    parser_create.set_defaults(func=create)

    # Train parser
    # ------------------------------------------

    parser_train = subparsers.add_parser(
        "train",
        help="Command to train model_0",
    )
    parser_train.set_defaults(func=train)

    # Evaluate parser
    # ------------------------------------------

    parser_evaluate = subparsers.add_parser(
        "evaluate",
        help="Command to evaluate an existing model_0"
    )
    parser_evaluate.set_defaults(func=evaluate)

    # Parse arguments
    # ------------------------------------------

    args = parser.parse_args()
    args.func(args)
