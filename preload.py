#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:09:58 2020

@author: minh
"""

import argparse
import re

import numpy as np
from tqdm import tqdm

import utils
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    x_train = np.load(utils.X_TRAIN_PATH, allow_pickle=True)
    y_train = np.load(utils.Y_TRAIN_PATH, allow_pickle=True) 
    
        
    index = []
    for i in range(len(x_train)):
        if x_train[i] =='':
          index.append(i)
    
    x_train_clean = np.delete(x_train, index) # 3247 emptpy cells
    y_train = np.delete(y_train, index)

    x_train1, x_train2, y_train1, y_train2 = train_test_split(
        x_train_clean, y_train, test_size=0.1, random_state=42)

    np.save(utils.X_TRAIN_PATH, x_train2)
    np.save(utils.Y_TRAIN_PATH, y_train2)