"""Folders tools."""

import os

DATASET_FOLDER = "dataset/"
RESULTS_FOLDER = "results/"
MODELS_FOLDER = "models/"


def create_dir(directory_path, delete_old=False):
    """Create directory.

    Args:
        directory_path (str): path of the directory to create
        delete_old (bool): set true if you want to remove old
                           directory (default false)
    """
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    else:
        if delete_old:
            os.rmdir(directory_path)
            os.mkdir(directory_path)
