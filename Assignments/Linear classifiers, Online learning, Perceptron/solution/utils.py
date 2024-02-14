import os
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logging.handlers import RotatingFileHandler


def init_logger(log_path, log_file_name):
    try:
        log_formatter = logging.Formatter("%(asctime)s  %(levelname)5.5s  %(filename)25s#%(lineno)3s  %(message)s")

        handler = RotatingFileHandler(
            filename=os.path.join(log_path, log_file_name),
            mode="a",
            maxBytes=(2 * 1024 * 1024),
            backupCount=4,
            encoding=None,
            delay=0,
        )

        handler.setFormatter(log_formatter)
        handler.setLevel(logging.DEBUG)
        log = logging.getLogger(log_file_name)
        log.setLevel(logging.DEBUG)

        log.addHandler(handler)
        log.debug("Start logging")
        return log

    except Exception as e:
        print("Failed to create logger: %s", str(e))
        return None


def get_key_of_max_value(map):
    max_key = ""
    max_val = float("-inf")

    for key, val in map.items():
        if val["accuracy"] >= max_val:
            max_val = val["accuracy"]
            max_key = key

    return max_key


def predict(example, weights, bias):
    value = np.dot(weights, example) + bias  # wT x + b
    return 1 if value >= 0 else -1


def test_accuracy(df, weights, bias):
    total = df.shape[0]
    correct_prediction = 0

    for _, row in df.iterrows():
        example = row.tolist()
        actual_label = example[0]  # y
        example = example[1:]  # x

        predicted_label = predict(example, weights, bias)

        if predicted_label == actual_label:
            correct_prediction += 1

    return correct_prediction / total


def prepare_train_test_folds(dfs):
    test_folds = []
    train_folds = []

    for i in range(len(dfs)):
        df_test = None
        df_train = []
        for df_index, df_element in enumerate(dfs):
            if i == df_index:
                df_test = df_element
                test_folds.append(df_test)
            else:
                df_train.append(df_element)

        df_train = pd.concat(df_train, ignore_index=True)  # List of folds are contacted and index is continuous.
        train_folds.append(df_train)

    return train_folds, test_folds


def initialize_weights_bias(rand_start, rand_end, feature_count):
    random_number = random.uniform(rand_start, rand_end)

    bias = random_number
    weights = []  # All weights and bias should be same.
    for _ in range(feature_count):
        weights.append(random_number)

    return weights, bias


def baseline_accuracy(test_df, dev_df):
    def get_accuracy(df, predict_label):
        total = df.shape[0]
        correct_prediction = 0

        for _, row in df.iterrows():
            example = row.tolist()
            actual_label = example[0]  # y

            if predict_label == actual_label:
                correct_prediction += 1

        return correct_prediction / total

    test_baseline_accuracy = 0
    test_df_pos = get_accuracy(df=test_df, predict_label=1)
    test_df_neg = get_accuracy(df=test_df, predict_label=-1)
    if test_df_pos > test_df_neg:
        print(f"Test dataset has more positive examples of accuracy: {test_df_pos}")
        test_baseline_accuracy = test_df_pos
    else:
        print(f"Test dataset has more negative examples of accuracy: {test_df_neg}")
        test_baseline_accuracy = test_df_neg

    dev_baseline_accuracy = 0
    dev_df_pos = get_accuracy(df=dev_df, predict_label=1)
    dev_df_neg = get_accuracy(df=dev_df, predict_label=-1)
    if dev_df_pos > dev_df_neg:
        print(f"Development dataset has more positive examples of accuracy: {dev_df_pos}")
        dev_baseline_accuracy = dev_df_pos
    else:
        print(f"Development dataset has more negative examples of accuracy: {dev_df_neg}")
        dev_baseline_accuracy = dev_df_neg

    return test_baseline_accuracy, dev_baseline_accuracy


def print_answer_1():
    print(
        "1. Briefly describe the design decisions that you have made in your implementation. (E.g, what programming language, how do you represent the vectors, etc.)"
    )
    print(
        """I used Python programming language, and datasets are represented using pandas library data frame object. The weights vector is stored as a Python list and bias as a float number. From the 5-fold cross-validation of 10 epochs, the hyperparameter that yields the best average cross-validation across all ten epochs is picked. The decided hyperparameter is used for online learning on the training dataset and tested on the dev dataset to get the best weights and bias. The best weight and bias are used to test the accuracy of the test dataset. I have used the same initial random weights and bias between -0.01 & 0.01 for cross-validation and online training. The dataset is shuffled using the random seed to produce consistent randomness.\n"""
    )


def plot_learning_curve(accuracies, baseline_accuracy, label):
    if not accuracies or type(accuracies) is not list:
        print(f"Can't plot learning curve. Invalid accuracies: {accuracies}")
        return

    fig = plt.figure()

    xticks = []
    epochs = []
    for i in range(len(accuracies)):
        epochs.append(i + 1)
        if i % 2 == 0:
            xticks.append(i)

    xticks.append(i + 1)

    max_epoch_accuracy = 0
    min_epoch_accuracy = 100
    for i in range(len(accuracies)):
        accuracies[i] = accuracies[i] * 100

        if accuracies[i] > max_epoch_accuracy:
            max_epoch_accuracy = accuracies[i]

        if accuracies[i] < min_epoch_accuracy:
            min_epoch_accuracy = accuracies[i]

    # Plot the data
    plt.plot(epochs, accuracies, marker="o", label="Dev Epoch Accuracy")
    plt.axhline(
        y=min_epoch_accuracy,
        color="orange",
        linestyle="dotted",
        label=f"Minimum Accuracy {round(min_epoch_accuracy, 3)}%",
    )
    plt.axhline(
        y=max_epoch_accuracy, color="g", linestyle="dotted", label=f"Maximum Accuracy {round(max_epoch_accuracy, 3)}%"
    )
    plt.axhline(
        y=baseline_accuracy, color="grey", linestyle="dotted", label=f"Baseline Accuracy {round(baseline_accuracy, 3)}%"
    )

    plt.xticks(xticks)
    plt.yticks([50, 60, 70, 80, 90])

    # Label the x-axis & y-axis
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")

    plt.legend()

    # Add title to graph
    plt.title(f"{label}'s Learning curve")

    # Save the figure
    fig.savefig("figs/{0}.png".format(label))
