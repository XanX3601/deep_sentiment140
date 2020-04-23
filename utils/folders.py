"""Folders tools."""

import os

DATASET_FOLDER = "dataset/"


def create_dir(directory_path, delete_old=False):
    """Create dir if not found.

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
