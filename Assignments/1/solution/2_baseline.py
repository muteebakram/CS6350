#!/usr/bin/python3

from data import Data

test_data_obj = Data(fpath="./data/test.csv")
train_data_obj = Data(fpath="./data/train.csv")


def get_most_common_label(data_obj, column=""):
    if not data_obj:
        print("Provide data object.")
        raise AttributeError

    if not column:
        print("Provide column name.")
        raise AttributeError

    data = data_obj.get_column(column)
    if not data.size:
        raise ValueError

    map = {}
    for d in data:
        if d not in map.keys():
            map[d] = 0

        map[d] += 1

    entries = 0
    most_common_label = ""
    for k, v in map.items():
        if v > entries:
            entries = v
            most_common_label = k

    # print(f"Most common label is {most_common_label} with {entries} entries.")
    return most_common_label


def calculate_accuracy(data_obj, predict=""):
    if not data_obj:
        print("Provide data object.")
        raise AttributeError

    if not predict:
        raise AttributeError

    data = data_obj.get_column("label")
    if not data.size:
        raise ValueError

    total = data.size
    correct_prediction = 0
    for d in data:
        if d == predict:
            correct_prediction += 1

    return (correct_prediction / total) * 100


most_common_label = get_most_common_label(train_data_obj, "label")
print(f"Test Accuracy : {calculate_accuracy(test_data_obj, most_common_label)}%")
print(f"Train Accuracy: {calculate_accuracy(train_data_obj, most_common_label)}%")
