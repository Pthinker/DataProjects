#!/usr/bin/env python

import pandas as pd
from sklearn import preprocessing
import numpy as np

import config

def get_csv_data(fpath):
    df = pd.read_csv(fpath, header=0, index_col=0)
    return df

def clean_data(data):
    imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer.fit(data)
    data = imputer.transform(data)
    data = preprocessing.scale(data)
    return data

def get_testing(fpath):
    test_data = get_csv_data(fpath)
    test_index = test_data.index.values
    test_data = clean_data(test_data)
    return test_index, test_data

def get_training(fpath):
    train = get_csv_data(fpath)

    cols = set(train.columns)
    cols.remove("loss")
    cols = list(cols)
    train_data = train[cols]
    train_label = train[["loss"]]

    train_data = clean_data(train_data)

    return train_data, train_label

def main():
    train_data, train_label = get_training(config.train_fpath)
    test_data = get_testing(config.test_fpath)

    for id, value in test_data.iterrows():
        print id, value

if __name__ == "__main__":
    main()

