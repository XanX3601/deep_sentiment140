from .folders import DATASET_FOLDER

TRAIN_CSV_PATH = "{}training.1600000.processed.noemoticon.csv".format(
    DATASET_FOLDER)
TEST_CSV_PATH = "{}testdata.manual.2009.06.14.csv".format(DATASET_FOLDER)

X_TRAIN_PATH = "{}x_train.npy".format(DATASET_FOLDER)
Y_TRAIN_PATH = "{}y_train.npy".format(DATASET_FOLDER)

X_TEST_PATH = "{}x_test.npy".format(DATASET_FOLDER)
Y_TEST_PATH = "{}y_test.npy".format(DATASET_FOLDER)

X_VALID_PATH = "{}x_valid.npy".format(DATASET_FOLDER)
Y_VALID_PATH = "{}y_valid.npy".format(DATASET_FOLDER)
