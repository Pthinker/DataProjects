#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import config
import utils


def plot_histogram(data):
    print len(data)
    max_value =  max(data)
    min_value =  min(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(data, 50, normed=1, facecolor='green', alpha=0.75)

    ax.set_xlabel('ACT')
    ax.set_ylabel('Probability')
    ax.set_xlim(min_value-0.5, max_value+0.5)
    ax.set_ylim(0, 1.1)
    ax.grid(True)

    plt.show()


def main():
    for i in range(1, 16):
        train_fpath = config.train_folder + "ACT%d_competition_training.csv" % i
        train_data, train_descriptors, train_molecules, act = utils.read_train(train_fpath)
        plot_histogram(act)


if __name__ == "__main__":
    main()
