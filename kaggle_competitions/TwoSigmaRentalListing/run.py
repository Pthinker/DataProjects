import numpy as np
import pandas as pd
import json



def main():
    train_df = pd.read_json("data/train.json")
    print train_df

if __name__ == "__main__":
    main()

