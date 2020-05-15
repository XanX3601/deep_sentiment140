"""Script to clean the dataset."""

import argparse
import re

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import trange

import utils


def cleaner(data):
    """Clean data.
    Args:
        data (str): string to clean
    Returns:
        data_clean (str): cleaned string
    """
    cleaning_regex = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    data_clean = re.sub(cleaning_regex, " ", data.lower()).strip()
    return data_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to clean Sentiment140 inputs data.")

    args = parser.parse_args()

    x_train = np.load(utils.X_TRAIN_PATH, allow_pickle=True)
    x_test = np.load(utils.X_TEST_PATH, allow_pickle=True)

    x_train_clean = []
    x_test_clean = []

    y_train = np.load(utils.Y_TRAIN_PATH)
    y_test = np.load(utils.Y_TEST_PATH)
    y_train_delete = []
    y_test_delete = []

    for i in trange(x_test.shape[0], desc="Cleaning test dataset"):
        clean_x_i = cleaner(x_test[i])

        # If string not empty
        if clean_x_i:
            x_test_clean.append(clean_x_i)
        else:
            y_test_delete.append(i)

    for i in trange(x_train.shape[0], desc="Cleaning train dataset"):
        clean_x_i = cleaner(x_train[i])

        # If string not empty
        if clean_x_i:
            x_train_clean.append(clean_x_i)
        else:
            y_train_delete.append(i)

    x_train_clean = np.array(x_train_clean)
    x_test_clean = np.array(x_test_clean)

    y_test = np.delete(y_test, y_test_delete)
    y_train = np.delete(y_train, y_train_delete)

    np.save(utils.X_TRAIN_PATH, x_train_clean)
    np.save(utils.X_TEST_PATH, x_test_clean)

    np.save(utils.Y_TRAIN_PATH, y_train)
    np.save(utils.Y_TEST_PATH, y_test)
