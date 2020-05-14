"""Script to clean the dataset."""

import argparse
import re

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

    for x_i in tqdm(x_test, desc="Cleaning test dataset"):
        clean_x_i = cleaner(x_i)

        # If string not empty
        if clean_x_i:
            x_test_clean.append(cleaner(x_i))

    for x_i in tqdm(x_train, desc="Cleaning train dataset"):
        clean_x_i = cleaner(x_i)

        # If string not empty
        if clean_x_i:
            x_train_clean.append(cleaner(x_i))

    x_train_clean = np.array(x_train_clean)
    x_test_clean = np.array(x_test_clean)

    np.save(utils.X_TRAIN_PATH, x_train_clean)
    np.save(utils.X_TEST_PATH, x_test_clean)
