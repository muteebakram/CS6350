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


def perceptron(df, weights, bias):
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

    return weights, bias


def cv_setup(train_df, test_df, learning_rate, epochs):
    # Initial random bias
    bias = random.uniform(rand_start, rand_end)

    # Initial random weights for each feature
    weights = []
    for _ in range(train_df.shape[1] - 1):
        weights.append(random.uniform(rand_start, rand_end))

    for epoch in range(epochs):
        # shuffle the whole data frame.
        train_df = train_df.sample(frac=1)

        weights, bias = perceptron(df=train_df, weights=weights, bias=bias)

        accuracy = test_accuracy(df=test_df, weights=weights, bias=bias)
        print(f"    Epoch: {epoch + 1:<3}  Accuracy: {accuracy}")


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
            print(f"\n  Cross validation {i}")
            cv_setup(train_df=train_folds[i], test_df=test_folds[i], learning_rate=learning_rate, epochs=10)

        print()

    # print("Online Training")
    # print(50 * "-")
    # for learning_rate in learning_rates:
    #     print(f"Learning rate: {learning_rate}")
    #     for i in range(len(dfs)):
    #         print(f"\n  Cross validation {i}")
    #         perceptron(train_df=train_folds[i], test_df=test_folds[i], learning_rate=learning_rate, epochs=10)

    #     print()
