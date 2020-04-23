"""Script to download Sentiment140 dataset."""

import argparse
import io
import zipfile

import requests

import utils

DATASET_URL = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

parser = argparse.ArgumentParser(
    description="Script to download Sentiment140 dataset. .csv files will be"
                "placed in {}".format(utils.DATASET_FOLDER))

if __name__ == "__main__":
    args = parser.parse_args()
    print("Downloading Sentiment140 dataset...")

    utils.create_dir(utils.DATASET_FOLDER, delete_old=True)

    r = requests.get(DATASET_URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(utils.DATASET_FOLDER)
    print("Done!")
