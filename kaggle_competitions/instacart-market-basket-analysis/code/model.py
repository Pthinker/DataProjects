import numpy as np


order_fpath = "../data/orders.csv"

def simpleModel():
    orders = np.genfromtxt(order_fpath, delimiter=",", skip_header=1)
    print orders.shape


def main():
    simpleModel()


if __name__ == "__main__":
    main()

