import os
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logging.handlers import RotatingFileHandler

random.seed(42)


def init_logger(log_path, log_file_name):
    try:
        log_formatter = logging.Formatter("%(asctime)s  %(levelname)5.5s  %(filename)10s#%(lineno)3s  %(message)s")

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


def predict(example, weights):
    value = np.dot(weights, example)  # wT x
    return 1 if value > 0 else -1


def test_accuracy(df, weights):
    total = df.shape[0]
    true_examples = 0
    correct_prediction = 0
    true_positive = 0  # classifier predicts as positive and are truly positive.
    false_positive = 0  # classifier predicts as positive, but are actually labeled negative
    false_negative = 0  # predicted as negative by your classifier, but are actually positive

    for _, row in df.iterrows():
        actual_label = int(row.tolist()[0])
        if actual_label > 0:
            true_examples += 1

        example = np.array(row.tolist()[1:])
        predicted_label = predict(example, weights)
        # print(f"actual_label: {actual_label}, predicted_label: {predicted_label}")
        if predicted_label == actual_label:
            correct_prediction += 1

        if predicted_label == 1 and actual_label == 1:
            true_positive += 1

        if predicted_label == 1 and actual_label == -1:
            false_positive += 1

        if predicted_label == -1 and actual_label == 1:
            false_negative += 1

    avg_accuracy = correct_prediction / total

    # Based on piazza post @332
    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        precision = 1

    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        recall = correct_prediction / true_examples

    try:
        f1_value = (2 * precision * recall) / (precision + recall)
        # print(f"precision {precision}, recall {recall} f1_value {f1_value}")
    except ZeroDivisionError:
        f1_value = 0
        # print(f"f1 value is zero precision {precision}, recall {recall} f1_value {f1_value}")

    return avg_accuracy, precision, recall, f1_value


def add_bias_to_df(df):
    df["bias"] = 1
    return df


def get_key_of_max_value(map):
    max_key = ""
    max_val = float("-inf")

    for key, val in map.items():
        if val >= max_val:
            max_val = val
            max_key = key

    return max_key


def plot_loss(loss, epochs, label):
    if not loss or type(loss) is not list:
        print(f"Can't plot loss curve. Invalid loss: {loss}")
        return

    fig = plt.figure()

    # Plot the data
    plt.plot(epochs, loss, label="Loss Epoch Curve")

    # plt.xticks(epochs)
    # plt.yticks([50, 60, 70, 80, 90])

    # Label the x-axis & y-axis
    plt.xlabel("Epochs")
    plt.ylabel("Loss Values")

    plt.legend()

    # Add title to graph
    plt.title(f"{label}'s Loss Epoch Curve")

    # Save the figure
    fig.savefig("figs/{0}.png".format(label))


def initialize_weights(rand_start, rand_end, feature_count):
    random_number = random.uniform(rand_start, rand_end)

    weights = []  # All weights and bias should be same.
    for _ in range(feature_count):
        weights.append(random_number)

    return weights
