#! /usr/bin/env python
"""
Merck Molecular Activity Challenge
"""

import sys
from os import path, listdir
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

import config
import utils


def r_squared(pred, obs):
    _, _, r, _, _= stats.linregress(pred, obs)
    return r**2


def randomforest(data, targets, num, fnum):
    """
    7:1205
    """
    model = RandomForestRegressor(n_estimators=num, verbose=0, oob_score=True, compute_importances=True, n_jobs=10, criterion="mse", max_features=fnum)
    model.fit(data, targets)
    return model


def write_pred(id, molecules, prediction):
    fh = open(config.result_folder + "rf_%d.csv" % id, "w")
    fh.write("MOLECULE,Prediction\n")
    for ind, pred in enumerate(prediction):
        fh.write("%s,%f\n" % (test_molecules[ind], pred))

    fh.close()


def main():
    id = int(sys.argv[1])

    train_fpath = config.train_folder + "ACT%d_competition_training.csv" % id
    #test_fpath = config.test_folder + "ACT%d_competition_test.csv" % id
    train_data, train_descriptors, train_molecules, train_targets = utils.read_train(train_fpath)
    #test_data, test_descriptors, test_molecules = utils.read_test(test_fpath)

    # Combine data and targets
    train_all = np.column_stack((train_data, train_targets))
    #np.random.shuffle(train_all)

    num = train_all.shape[0] * 0.8
    train = train_all[0:num, ]
    test = train_all[num:, ]
    train_x = train[:, 0:(train.shape[1]-1)]
    train_y = train[:, -1]
    test_x = test[:, 0:(test.shape[1]-1)]
    test_y = test[:, -1]

    for i in range(100, 5001, 200):
        model = randomforest(train_x, train_y, 100, i)
        prediction = model.predict(test_x)

        print "%d Rsquare: %f" % (i, r_squared(prediction, test_y))


if __name__ == "__main__":
    main()

