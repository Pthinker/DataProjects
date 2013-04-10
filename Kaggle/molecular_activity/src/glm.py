#!/usr/bin/env python

from scipy import stats
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import linear_model

import config
import utils


def rsquare(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value**2


def lasso(train, targets):
    cv_model = linear_model.LassoCV(cv=10)
    cv_model.fit(train, targets)
    best_alpha = cv_model.alpha_

    model = linear_model.Lasso(alpha=best_alpha)
    model.fit(train, targets)
    return model


def ridge(train, targets):
    model = linear_model.RidgeCV(alphas=[100, 300, 600], cv=5)
    model.fit(train, targets)
    return model    


def main():
    for i in range(7, 8):
        fh = open(config.result_folder + "lasso.csv", "w")
        fh.write("MOLECULE,Prediction\n")

        train_fpath = config.train_folder + "ACT%d_competition_training.csv" % i
        test_fpath = config.test_folder + "ACT%d_competition_test.csv" % i 
        train_data, train_descriptors, train_molecules, train_targets = utils.read_train(train_fpath)
        test_data, test_descriptors, test_molecules = utils.read_test(test_fpath)

        model = lasso(train_data, train_targets)

        results = model.predict(test_data)

        for ind, result in enumerate(results):
            fh.write("%s,%f\n" % (test_molecules[ind], result))

        fh.close()


if __name__ == "__main__":
    main()

