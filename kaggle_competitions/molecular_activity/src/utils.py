#!/usr/bin/env python

import numpy as np

import config


def read_train(fpath):
    fh = open(fpath)

    header = fh.readline().strip()
    arr = header.split(",")
    descriptors = arr[2:]

    molecules = []
    act = []
    matrix = []
    for line in fh:
        line = line.strip()
        arr = line.split(",")
        molecules.append(arr.pop(0))
        act.append(float(arr.pop(0)))
        matrix.append(map(int, arr))
    return np.array(matrix), descriptors, molecules, np.array(act)


def read_test(fpath):
    fh = open(fpath)

    header = fh.readline().strip()
    arr = header.split(",")
    descriptors = arr[1:]

    molecules = []
    matrix = []
    for line in fh:
        line = line.strip()
        arr = line.split(",")
        molecules.append(arr.pop(0))
        matrix.append(map(int, arr))
    return np.array(matrix), descriptors, molecules


def main():
    pass


if __name__ == "__main__":
    main()
