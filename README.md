# deep_sentiment140

Deep learning approaches for Sentiment140 dataset.

## Installation

To get started, clone the project.

```shell
git clone https://github.com/XanX3601/deep_sentiment140.git
cd deep_sentiment140
```

## Prerequisites

[Python](https://www.python.org/) 3.6 or higher is required.

All Python requirements can be found in `requirements.txt` and can be installed as follow.

```shell
pip install -r requirements.txt
```

## Dataset

The dataset is Sentiment140. You can find the official website [here](http://help.sentiment140.com/for-students).

- To get data, you first need to download raw `.csv` files with the following command: `python download_data.py`.
- Then you need to create `numpy` files for train and test datasets with the following command: `python create_datasets.py`.

## Models

To answer the sentiment analysis problem, different models are proposed. In `models/model_name` are stored the different models. In `results/model_name` are stored results and logs. Each model works as follow. For each command, you can add `-h` to get more info.

- To create the model, execute:

```shell
python model_name.py create
```

- To train the model, execute:

```shell
python model_name.py train
```

- To evaluate the model, execute:

```shell
python model_name.py evaluate
```
