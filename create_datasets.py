"""Script to create Sentiment140 datasets."""

import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import utils

parser = argparse.ArgumentParser(
    description="Script to create Sentiment140 train and test datasets. The "
                "following files will be created: {}, {}, {} and {}".format(
                    utils.X_TRAIN_PATH,
                    utils.Y_TRAIN_PATH,
                    utils.X_TEST_PATH,
                    utils.Y_TEST_PATH,
                    utils.X_VALID_PATH,
                    utils.Y_VALID_PATH,))

if __name__ == "__main__":
    args = parser.parse_args()
    print("Creating numpy files...")

    names = ["polarity", "id", "date", "query", "user", "text"]

    csv_train = pd.read_csv(utils.TRAIN_CSV_PATH,
                            encoding="latin-1", names=names)
    csv_valid = pd.read_csv(utils.TEST_CSV_PATH,
                           encoding="latin-1", names=names)

    x_data = csv_train["text"].to_numpy()    
    x_valid_data = csv_valid["text"].to_numpy()

    y_data = csv_train["polarity"].to_numpy().astype(int)
    y_valid_data = csv_valid["polarity"].to_numpy().astype(int)
    

    (x_train_data, x_test_data, y_train_data, y_test_data) = train_test_split(x_data, y_data, test_size=0.15, random_state=42)
    
    print("Number of train data: {}".format(len(x_train_data)))
    print("Number of test data: {}".format(len(x_test_data)))
    print("Number of validation data: {}".format(len(x_valid_data)))

    np.save(utils.X_TRAIN_PATH, x_train_data)
    np.save(utils.Y_TRAIN_PATH, y_train_data)
    np.save(utils.X_TEST_PATH, x_test_data)
    np.save(utils.Y_TEST_PATH, y_test_data)
    
    np.save(utils.X_VALID_PATH, x_valid_data)
    np.save(utils.Y_VALID_PATH, y_valid_data)

    print("Done!")
