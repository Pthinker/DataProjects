import os
import csv
import pandas
import pickle

import config


def identity(x):
    return x

converters = { "FullDescription" : identity, 
               "Title": identity,
               "LocationRaw": identity,
               "LocationNormalized": identity
             }


def get_train_df():
    return pandas.read_csv(config.TRAIN_FPATH, converters=converters)


def get_valid_df():
    return pandas.read_csv(config.VALID_FPATH, converters=converters)


def save_model(model):
    pickle.dump(model, open(config.MODEL_FPATH, "w"))


def load_model():
    return pickle.load(open(config.MODEL_FPATH))


def write_submission(predictions):
    fh = csv.writer(open(config.PRED_FPATH, "w"), lineterminator="\n")
    valid_df = get_valid_df()
    rows = [row for row in zip(valid_df["Id"], predictions.flatten())]
    fh.writerow(("Id", "SalaryNormalized"))
    fh.writerows(rows)


def main():
    train_df = get_train_df()
    print train_df.shape


if __name__ == "__main__":
    main()
