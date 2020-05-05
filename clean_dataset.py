#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:53:16 2020

@author: minh
"""
import argparse

from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer
import utils
import numpy as np

def cleaner(data):

    #Remove HTML Code
    data_clean = BeautifulSoup(data, features='html')
    data_clean = data_clean.get_text()

    #Remove URL http
    data_clean = re.sub('https?://[A-Za-z0-9./]+','', data_clean)

    #Remove mention @
    data_clean = re.sub(r'@[A-Za-z0-9]+','', data_clean)

    #Remove hastag #, or other informations, keep only letter
    data_clean = re.sub("[^a-zA-Z]", " ", data_clean)

    #Lowercase letters
    data_clean = data_clean.lower()

    #Remove unnecessary spaces
    tok = WordPunctTokenizer()
    token_clean = tok.tokenize(data_clean)
    token_clean = (" ".join(token_clean)).strip()

    return token_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to create Sentiment140 train and test datasets. "
                    "The following files will be created: {}, {}, {} and {}"
                    .format(utils.X_TRAIN_PATH, utils.Y_TRAIN_PATH,
                            utils.X_TEST_PATH, utils.Y_TEST_PATH,))

    args = parser.parse_args()
    
    print("Loading numpy files...")

    x_test = np.load(utils.X_TRAIN_PATH, allow_pickle=True, fix_imports=True, encoding='ASCII')
    x_train = np.load(utils.X_TEST_PATH, allow_pickle=True, fix_imports=True, encoding='ASCII')

    x_train_clean = []    
    x_test_clean = []
    
    for x_i in x_test:
        x_test_clean.append(cleaner(x_i))
        
    print("Finish cleaning the test dataset!...")
        
    for x_i in x_train:
        x_train_clean.append(cleaner(x_i))
        
    print("Finish cleaning the train dataset!")

    np.save(utils.X_TRAIN_PATH, x_train_clean)
    #np.save(utils.Y_TRAIN_PATH, y_train_data)
    np.save(utils.X_TEST_PATH, x_test_clean)
    #np.save(utils.Y_TEST_PATH, y_test_data)
    

    print("Done!")