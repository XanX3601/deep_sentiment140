# deep_sentiment140

Deep learning approaches for Sentiment140 dataset.

## Prerequisites

You need to have Python 3.6 (at least) installed on your system. To run this project, commands are the following.

```shell
git clone https://github.com/XanX3601/deep_sentiment140.git
cd deep_sentiment140
pip install -r requirements.txt
```

## Dataset

The dataset is Sentiment140. You can find the official website [here](http://help.sentiment140.com/for-students).

- To get data, you first need to download raw `.csv` files with the following command: `python download_data.py`.
- Then you need to create `numpy` files for train and test datasets with the following command: `python create_datasets.py`.
