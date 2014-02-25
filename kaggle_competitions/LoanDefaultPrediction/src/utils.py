#!/usr/bin/env python

import pandas as pd
import config

def get_csv_data(fpath):
    df = pd.read_csv(fpath, header=0, index_col=0)
    return df

def main():
    train = get_csv_data(config.train_fpath)
    print train.shape

    cols = set(train.columns)
    cols.remove("loss")
    cols = list(cols)
    train_data = train[cols]
    train_label = train[["loss"]]

    print train_data.shape
    print train_label.shape

if __name__ == "__main__":
    main()

