import os

# data
data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data")

train_fpath = os.path.join(data_folder, "train_v2.csv")
test_fpath = os.path.join(data_folder, "test_v2.csv")


result_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "result")
