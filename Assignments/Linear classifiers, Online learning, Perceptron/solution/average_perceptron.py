import random
import numpy as np
import pandas as pd

rand_start = -0.01
rand_end = 0.01

learning_rates = [1, 0.1, 0.01]

df0 = pd.read_csv("./hw2_data/CVSplits/train0.csv")
df1 = pd.read_csv("./hw2_data/CVSplits/train1.csv")
df2 = pd.read_csv("./hw2_data/CVSplits/train2.csv")
df3 = pd.read_csv("./hw2_data/CVSplits/train3.csv")
df4 = pd.read_csv("./hw2_data/CVSplits/train4.csv")

dfs = [df0, df1, df2, df3, df4]

train_df = pd.read_csv("./hw2_data/diabetes.train.csv")
dev_df = pd.read_csv("./hw2_data/diabetes.dev.csv")
test_df = pd.read_csv("./hw2_data/diabetes.test.csv")


def predict(example, weights, bias):
    value = np.dot(weights, example) + bias  # wT x + b
    return 1 if value >= 0 else -1


def test_accuracy(df, weights, bias):
    total = df.shape[0]
    correct_prediction = 0

    for index, row in df.iterrows():
        example = row.tolist()
        actual_label = example[0]  # y
        example = example[1:]  # x

        predicted_label = predict(example, weights, bias)

        if predicted_label == actual_label:
            correct_prediction += 1

    return correct_prediction / total


def perceptron(df, learning_rate, weights, bias, avg_weight, avg_bias):
    for index, row in df.iterrows():
        example = row.tolist()
        actual_label = example[0]  # y
        example = example[1:]  # x

        # y(wT x + b)
        value = actual_label * (np.dot(weights, example) + bias)

        # update
        if value < 0:
            bias += learning_rate * actual_label
            for index, w in enumerate(weights):
                w += learning_rate * actual_label * example[index]

        # average update
        avg_bias += bias
        for index, w in enumerate(weights):
            avg_weight[index] += w

    return weights, bias, avg_weight, avg_bias


def cv_setup(train_fold_df, test_fold_df, learning_rate, epochs):
    # Initial random bias
    bias = random.uniform(rand_start, rand_end)

    # Initial random weights for each feature
    weights = []
    for _ in range(train_fold_df.shape[1] - 1):
        weights.append(random.uniform(rand_start, rand_end))

    # Initialize avg to the initial values.
    avg_bias = bias
    avg_weight = weights

    for epoch in range(epochs):
        # shuffle the whole data frame.
        train_fold_df = train_fold_df.sample(frac=1)

        # each epoch produce new weight and bias which is input to next epoch.
        weights, bias, avg_weight, avg_bias = perceptron(
            df=train_fold_df,
            learning_rate=learning_rate,
            weights=weights,
            bias=bias,
            avg_weight=avg_weight,
            avg_bias=avg_bias,
        )

        accuracy = test_accuracy(df=test_fold_df, weights=avg_weight, bias=avg_bias)
        print(f"    Epoch: {epoch + 1:>2}    Accuracy: {accuracy}")


def online_setup(train_df, dev_df, test_df, learning_rate, epochs):
    # Initial random bias
    bias = random.uniform(rand_start, rand_end)

    # Initial random weights for each feature
    weights = []
    for _ in range(train_df.shape[1] - 1):
        weights.append(random.uniform(rand_start, rand_end))

    # Initialize avg to the initial values.
    avg_bias = bias
    avg_weight = weights

    best_bias = 0
    best_weights = 0
    best_epoch = 0
    best_accuracy = 0
    for epoch in range(epochs):
        # shuffle the whole data frame.
        train_df = train_df.sample(frac=1)

        # each epoch produce new weight and bias which is input to next epoch.
        weights, bias, avg_weight, avg_bias = perceptron(
            df=train_df,
            learning_rate=learning_rate,
            weights=weights,
            bias=bias,
            avg_weight=avg_weight,
            avg_bias=avg_bias,
        )

        accuracy = test_accuracy(df=dev_df, weights=avg_weight, bias=avg_bias)
        print(f"  Epoch: {epoch + 1:>2}    Accuracy: {accuracy}")

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch

            best_bias = avg_bias
            best_weights = avg_weight

    test_data_accuracy = test_accuracy(df=test_df, weights=best_weights, bias=best_bias)
    print(f"\n  Best Epoch: {best_epoch + 1:>2}    Test Accuracy: {test_data_accuracy}")


def prepare_train_test_folds():
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


if __name__ == "__main__":
    train_folds, test_folds = prepare_train_test_folds()
    # print(train_folds[0])

    print("Cross Validation Training")
    print(50 * "-")
    for learning_rate in learning_rates:
        print(f"Learning rate: {learning_rate}")
        for i in range(len(dfs)):
            print(f"  Cross validation {i}")
            cv_setup(train_fold_df=train_folds[i], test_fold_df=test_folds[i], learning_rate=learning_rate, epochs=10)
            print()

        print()

    print("Online Training")
    print(50 * "-")
    for learning_rate in learning_rates:
        print(f"Learning rate: {learning_rate}")
        online_setup(train_df=train_df, dev_df=dev_df, test_df=test_df, learning_rate=learning_rate, epochs=20)
        print()
