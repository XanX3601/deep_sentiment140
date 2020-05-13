"""Script to create Sentiment140 datasets."""

import argparse

import numpy as np
import pandas as pd

import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to create Sentiment140 train and test datasets. "
                    "The following files will be created: {}, {}, {} and {}"
                    .format(utils.X_TRAIN_PATH, utils.Y_TRAIN_PATH,
                            utils.X_TEST_PATH, utils.Y_TEST_PATH,))

    args = parser.parse_args()
    print("Creating numpy files...")

    names = ["polarity", "id", "date", "query", "user", "text"]

    csv_train = pd.read_csv(utils.TRAIN_CSV_PATH,
                            encoding="latin-1", names=names)
    csv_test = pd.read_csv(utils.TEST_CSV_PATH,
                           encoding="latin-1", names=names)

    x_train_data = csv_train["text"].to_numpy()
    x_test_data = csv_test["text"].to_numpy()

    y_train_data = csv_train["polarity"].to_numpy().astype(int)
    y_test_data = csv_test["polarity"].to_numpy().astype(int)

    np.save(utils.X_TRAIN_PATH, x_train_data)
    np.save(utils.Y_TRAIN_PATH, y_train_data)
    np.save(utils.X_TEST_PATH, x_test_data)
    np.save(utils.Y_TEST_PATH, y_test_data)

    print("Done!")