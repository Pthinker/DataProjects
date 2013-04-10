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


class processing():
    """ Loops over the files and appends predictions.

    Assumes project structure is:
        merck/
            src/
                model.py
            data/
                train/
                    ACT1_competition_training.csv
                    ...
                    ACT15_competition_training.csv
                test/
                    ACT1_competition_test.csv
                    ...
                    ACT15_competition_test.csv
            submission/

    Example:
        >>> p = processing(prediction_fname='glm.csv')
        >>> for train_x, train_y, test_x, test_labels in p:
                model.fit(train_x, train_y)
                prediction = model.predict(test_x)
                p.append_prediction(prediction, test_labels)
    """
    def __init__(self, prediction_fname='prediction.csv'):
        """ Defines locations of project management, iteration params.
        """
        current_path = path.dirname(path.abspath(__file__))
        parent_dir = path.dirname(current_path)
        self.prediction_fname = prediction_fname
        self.data_dir = path.join(parent_dir, 'data')
        self.sub_dir = path.join(parent_dir, 'submission')
        self.out_path = path.join(self.sub_dir, self.prediction_fname)
        self.fail_if_pred_fname_exists()
        self.get_fnames()
        self.start = 0
        self.stop = len(self.training_fnames) - 1

    def __iter__(self):
        return self

    def next(self):
        if self.start > self.stop:
            StopIteration
        elif self.start >= len(self.training_fnames) - 1:
            StopIteration
        else:
            cur = self.start
            self.start += 1
            train_x, train_y = self.get_train(cur)
            test_x, test_labels = self.get_test(cur)
            return train_x, train_y,\
                    test_x, test_labels

    def get_fnames(self):
        """ Makes list objects of training and test filenames.
        """
        #  Takes string and returns a integer of the numbers in the string.
        fnk = lambda a: int(''.join([i for i in a if i.isdigit()]))
        self.training_fnames = listdir(path.join(self.data_dir,'train'))
        self.training_fnames.sort(key=fnk)
        self.testing_fnames = listdir(path.join(self.data_dir,'test'))
        self.testing_fnames.sort(key=fnk)

    def get_train(self,numero):
        train_path = path.join(self.data_dir,'train')
        train_df = pandas.read_csv(
                path.join(train_path,self.training_fnames[numero]))
        print("Current file: %s" % self.training_fnames[numero])
        del train_df['MOLECULE']
        train_y = np.array(train_df['Act'].tolist())
        del train_df['Act']
        train_x = train_df.as_matrix()
        return train_x, train_y

    def get_test(self,numero):
        test_path = path.join(self.data_dir,'test')
        test_df = pandas.read_csv(
                path.join(test_path,self.testing_fnames[numero]))
        test_labels = test_df['MOLECULE'].tolist()
        del test_df['MOLECULE']
        test_x = test_df.as_matrix()
        return test_x, test_labels

    def fail_if_pred_fname_exists(self):
        if path.isfile(self.prediction_fname):
            raise Exception(
                "The output filename passed to processing already exists.")
        else:
            out_handle = open(self.out_path,'w')
            out_handle.write('"MOLECULE","Prediction"\n')
            out_handle.close()

    def append_prediction(self, prediction, test_labels):
        out_handle = open(self.out_path,'a')
        for label, pred in zip(test_labels, prediction):
            out_handle.write('"%s",' % label)
            out_handle.write('%3.10f\n' % pred)
        out_handle.close()


def r_squared(pred, obs):
    _, _, r, _, _= stats.linregress(pred, obs)
    return r**2


def randomforest(data, targets, num):
    """
    2:5277
    3:1203
    4:1506
    5:1274
    7:1205
    8:1603
    9:1930
    10:2190
    11:735
    12:1270
    13:1098
    14:1345
    15:1552
    """
    model = RandomForestRegressor(n_estimators=num, verbose=0, oob_score=True, compute_importances=True, n_jobs=10, max_features=5277)
    model.fit(data, targets)
    return model


def write_pred(id, molecules, prediction):
    fh = open(config.result_folder + "rf_%d.csv"%id, "w")
    fh.write("MOLECULE,Prediction\n")
    for ind, pred in enumerate(prediction):
        fh.write("%s,%f\n" % (molecules[ind], pred))

    fh.close()


def main():
    id = int(sys.argv[1])

    train_fpath = config.train_folder + "ACT%d_competition_training.csv" % id
    test_fpath = config.test_folder + "ACT%d_competition_test.csv" % id
    train_data, train_descriptors, train_molecules, train_targets = utils.read_train(train_fpath)
    test_data, test_descriptors, test_molecules = utils.read_test(test_fpath)

    model = randomforest(train_data, train_targets, 1000)

    #for ind, fi in enumerate(model.feature_importances_):

    prediction = model.predict(test_data)

    write_pred(id, test_molecules, prediction)


if __name__ == "__main__":
    main()

