#!/usr/bin/env python

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

import utils
import config

def randomForest(train_data, target, test_data):
    model = RandomForestRegressor(n_estimators=10, min_samples_split=2, n_jobs=4)
    model.fit(train_data, target)
    predictions = model.predict(test_data)
    
    print predictions

def main():
    train_data, train_label = utils.get_training(config.train_fpath)
    test_data = utils.get_testing(config.test_fpath)
    
    randomForest(train_data, train_label, test_data)


if __name__ == "__main__":
    main()
