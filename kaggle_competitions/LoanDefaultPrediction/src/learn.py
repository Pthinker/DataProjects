#!/usr/bin/env python

import os
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

import utils
import config

def randomForest(train_data, target, test_data):
    model = RandomForestRegressor(n_estimators=10, min_samples_split=2, n_jobs=4)
    model.fit(train_data, target)
    predictions = model.predict(test_data)
 
    return predictions

def write_result(test_index, predictions, fname="result.csv"):
    with open(os.path.join(config.result_folder, fname)) as fh:
        fh.write("id,loss\n")
        for ind, val in enumerate(test_index):
            fh.write(val + "," + str(int(prediction[ind])) + "\n")

def main():
    train_data, train_label = utils.get_training(config.train_fpath)
    test_index, test_data = utils.get_testing(config.test_fpath)
    
    predictions = randomForest(train_data, train_label, test_data)
    write_result(test_index, predictions)

if __name__ == "__main__":
    main()
