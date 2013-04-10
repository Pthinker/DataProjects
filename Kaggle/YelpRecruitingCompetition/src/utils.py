import os
import json

import config


def read_json_data(fpath):
    """
    Open JSON format file (each line is json format) and return a list of json dict
    """
    fh = open(fpath, 'r')
    json_data = []
    for line in fh:
        data = json.loads(line)
        json_data.append(data)
    return json_data


def main():
    read_json_data(config.TEST_USER_FPATH)


if __name__ == "__main__":
    main()
