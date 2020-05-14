import argparse
import re

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import utils
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer


def cleaner(data):
    """Clean data.
    Args:
        data (str): string to clean
    Returns:
        token_clean (str): cleaned string
    """
    # Remove HTML Code
    data_clean = BeautifulSoup(data, features="lxml")
    data_clean = data_clean.get_text()

    # Remove URL http
    data_clean = re.sub("https?://[A-Za-z0-9./]+", "", data_clean)

    # Remove mention @
    data_clean = re.sub(r"@[A-Za-z0-9]+", "", data_clean)

    # remove \xef\xbf\xbd (UTF-8 BOM)
    try:
        data_clean = data_clean.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        data_clean = data_clean

    # Remove hastag #, and other informations, keep only letter
    data_clean = re.sub("[^a-zA-Z]", " ", data_clean)

    # Lowercase letters
    data_clean = data_clean.lower()

    # Remove unnecessary spaces
    tok = WordPunctTokenizer()
    token_clean = tok.tokenize(data_clean)
    token_clean = (" ".join(token_clean)).strip()

    return token_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to clean Sentiment140 inputs data.")

    args = parser.parse_args()

    x_train = np.load(utils.X_TRAIN_PATH, allow_pickle=True)
    x_test = np.load(utils.X_TEST_PATH, allow_pickle=True)

    x_train_clean = []
    x_test_clean = []

    for x_i in tqdm(x_test, desc="Cleaning test dataset"):
        x_test_clean.append(cleaner(x_i))

    for x_i in tqdm(x_train, desc="Cleaning train dataset"):
        x_train_clean.append(cleaner(x_i))

    x_train_clean = np.array(x_train_clean)
    x_test_clean = np.array(x_test_clean)

    np.save(utils.X_TRAIN_PATH, x_train_clean)
    np.save(utils.X_TEST_PATH, x_test_clean)
