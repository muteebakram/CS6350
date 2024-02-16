import numpy as np
import pandas as pd
from utils import (
    test_accuracy,
    prepare_train_test_folds,
    initialize_weights_bias,
    baseline_accuracy,
    init_logger,
    plot_learning_curve,
    get_key_of_max_value,
    print_answer_1,
)


log = None

perceptron_type = "Average Perceptron"

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


def perceptron(df, learning_rate, weights, bias, epoch, avg_weights, avg_bias):
    update_count = 0
    example_count = 0

    for _, row in df.iterrows():
        example_count += 1

        example = row.tolist()
        actual_label = example[0]  # y
        example = example[1:]  # x

        # y(wT x + b)
        value = actual_label * (np.dot(weights, example) + bias)

        # update
        if value < 0:
            update_count += 1
            bias += learning_rate * actual_label
            for index in range(len(weights)):
                weights[index] += learning_rate * actual_label * example[index]

        avg_bias += bias
        for index in range(len(weights)):
            avg_weights[index] += weights[index]

    return weights, bias, avg_weights, avg_bias, update_count


def cv_setup(train_fold_df, test_fold_df, learning_rate, weights, bias, epochs):
    # Initialize avg to the initial values.
    avg_bias = bias
    avg_weight = weights[:]  # Copy the list because list is mutable.

    for epoch in range(epochs):
        # shuffle the whole data frame.
        train_fold_df = train_fold_df.sample(frac=1, random_state=1)

        # each epoch produce new weight and bias which is input to next epoch.
        weights, bias, avg_weight, avg_bias, _ = perceptron(
            df=train_fold_df,
            learning_rate=learning_rate,
            weights=weights,
            bias=bias,
            epoch=epoch,
            avg_weights=avg_weight,
            avg_bias=avg_bias,
        )

    accuracy = test_accuracy(df=test_fold_df, weights=avg_weight, bias=avg_bias)
    return accuracy


def online_setup(train_df, dev_df, learning_rate, weights, bias, epochs):
    # Initialize avg to the initial values.
    avg_bias = bias
    avg_weight = weights[:]  # Copy the list because list is mutable.
    avg_weight, avg_bias = initialize_weights_bias(
        rand_start=rand_start, rand_end=rand_end, feature_count=train_df.shape[1] - 1
    )

    best_epoch = 0
    best_accuracy = 0
    total_update_counts = 0
    dev_accuracies = []

    log.debug(f"Learning rate: {learning_rate}")
    for epoch in range(epochs):
        # shuffle the whole data frame.
        train_df = train_df.sample(frac=1, random_state=1)

        # each epoch produce new weight and bias which is input to next epoch.
        weights, bias, avg_weight, avg_bias, update_count = perceptron(
            df=train_df,
            learning_rate=learning_rate,
            weights=weights,
            bias=bias,
            epoch=epoch,
            avg_weights=avg_weight,
            avg_bias=avg_bias,
        )
        total_update_counts += update_count

        accuracy = test_accuracy(df=dev_df, weights=weights, bias=bias)
        log.debug(f"  Epoch: {epoch + 1:>2}    Accuracy: {accuracy}")
        print(f"   Epoch: {epoch + 1:>2}    Dev Accuracy: {accuracy}")
        dev_accuracies.append(accuracy)

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch

    log.debug(f"  Best Epoch: {best_epoch + 1:>2}    Dev Accuracy: {best_accuracy}")
    print(
        f"\nc. The total number of updates the learning algorithm (learning rate {learning_rate}) performs on the training set: {total_update_counts}"
    )
    return avg_weight, avg_bias, dev_accuracies


if __name__ == "__main__":
    log = init_logger(log_path="./logs", log_file_name=f"{perceptron_type.replace(' ', '_').lower()}.log")

    train_folds, test_folds = prepare_train_test_folds(dfs=dfs)

    # I plan to use same initial setting for CV and online learning.
    initial_weights, initial_bias = initialize_weights_bias(
        rand_start=rand_start, rand_end=rand_end, feature_count=train_df.shape[1] - 1
    )
    log.debug(f"Initial Bias: {initial_bias}")
    log.debug(f"Initial Weights: {initial_weights}")

    print_answer_1()

    print("2. Majority Baseline")
    _, dev_baseline_accuracy = baseline_accuracy(test_df=test_df, dev_df=dev_df)

    print("\n3. Variants Result")
    log.debug(50 * "-")
    log.debug("Cross Validation Training")

    # Key is the learning rate and values list of accuracy.
    # For each key, the values in list represent the best cv accuracy of that fold for given learning rate.
    all_cv_accuracies_dict = {}

    for learning_rate in learning_rates:
        log.debug(f"Learning rate: {learning_rate}")
        if learning_rate not in all_cv_accuracies_dict:
            all_cv_accuracies_dict[learning_rate] = []

        for i in range(len(dfs)):
            accuracy = cv_setup(
                train_fold_df=train_folds[i],
                test_fold_df=test_folds[i],
                learning_rate=learning_rate,
                weights=initial_weights,
                bias=initial_bias,
                epochs=10,
            )
            all_cv_accuracies_dict[learning_rate].append(accuracy)
            log.debug(f"  Cross validation (Fold {i}) Accuracy: {accuracy}")

        log.debug("")

    log.debug(f"All CV accuracy dictionary: {all_cv_accuracies_dict}")

    best_lr_parameter = 0
    best_hyper_parameter = 0
    hyper_parameter_setting = {}

    for learning_rate, cv_accuracies in all_cv_accuracies_dict.items():
        avg_fold_accuracy = 0
        hyper_parameter = f"learning rate {learning_rate}"
        hyper_parameter_setting[hyper_parameter] = {"learning_rate": 0}

        for accuracy in cv_accuracies:
            avg_fold_accuracy += accuracy

        hyper_parameter_setting[hyper_parameter]["learning_rate"] = learning_rate
        hyper_parameter_setting[hyper_parameter]["accuracy"] = avg_fold_accuracy / len(cv_accuracies)

        log.debug(f"Average accuracy of all folds with learning rate {learning_rate} is {avg_fold_accuracy / len(dfs)}")

    best_hyper_parameter = get_key_of_max_value(hyper_parameter_setting)
    best_lr_parameter = hyper_parameter_setting[best_hyper_parameter]["learning_rate"]
    best_cv_avg_accuracy = hyper_parameter_setting[best_hyper_parameter]["accuracy"]

    log.debug(f"All hyper parameter setting dictionary: {hyper_parameter_setting}")

    print(f"a. The best hyper-parameters is {best_hyper_parameter}")
    print(
        f"\nb. The cross-validation accuracy for the best hyperparameter ({best_hyper_parameter}) is {best_cv_avg_accuracy}"
    )

    log.debug(50 * "-")
    log.debug("Online Training")

    print(f"\nd. Development set accuracy for best hyper parameter ({best_hyper_parameter})")
    best_weights, best_bias, dev_accuracies = online_setup(
        train_df=train_df,
        dev_df=dev_df,
        learning_rate=best_lr_parameter,
        weights=initial_weights,
        bias=initial_bias,
        epochs=20,
    )

    test_data_accuracy = test_accuracy(df=test_df, weights=best_weights, bias=best_bias)
    log.debug(f"Test set accuracy for best hyper parameter ({best_hyper_parameter}) is {test_data_accuracy}")
    print(f"\ne. Test set accuracy for best hyper parameter ({best_hyper_parameter}) is {test_data_accuracy}")

    plot_learning_curve(accuracies=dev_accuracies, baseline_accuracy=dev_baseline_accuracy * 100, label=perceptron_type)
    print(
        f"\nf. Plot a learning curve where the x axis is the epoch id and the y axis is the dev set accuracy using the classifier. Check figure './figs/{perceptron_type}.png'"
    )
