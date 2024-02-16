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

perceptron_type = "Margin Perceptron"

rand_start = -0.01
rand_end = 0.01

margins = [1, 0.1, 0.01]
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


def perceptron(df, margin, learning_rate, weights, bias):
    update_count = 0
    for _, row in df.iterrows():
        example = row.tolist()
        actual_label = example[0]  # y
        example = example[1:]  # x

        # y(wT x + b)
        value = actual_label * (np.dot(weights, example) + bias)

        # update
        if value < margin:
            update_count += 1
            bias += learning_rate * actual_label
            for index in range(len(weights)):
                # w = w + r * y * x
                weights[index] += learning_rate * actual_label * example[index]

    return weights, bias, update_count


def cv_setup(train_fold_df, test_fold_df, margin, learning_rate, weights, bias, epochs):
    for epoch in range(epochs):
        # shuffle the whole data frame.
        train_fold_df = train_fold_df.sample(frac=1, random_state=1).reset_index(drop=True)

        new_learning_rate = learning_rate / (1 + epoch)

        # each epoch produce new weight and bias which is input to next epoch.
        weights, bias, _ = perceptron(
            df=train_fold_df, margin=margin, learning_rate=new_learning_rate, weights=weights, bias=bias
        )

    accuracy = test_accuracy(df=test_fold_df, weights=weights, bias=bias)
    return accuracy


def online_setup(train_df, dev_df, margin, learning_rate, weights, bias, epochs):
    best_bias = 0
    best_weights = 0
    best_epoch = 0
    best_accuracy = 0
    total_update_counts = 0
    dev_accuracies = []

    log.debug(f"Learning rate: {learning_rate}")
    for epoch in range(epochs):
        # shuffle the whole data frame.
        train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)

        new_learning_rate = learning_rate / (1 + epoch)

        # each epoch produce new weight and bias which is input to next epoch.
        weights, bias, update_count = perceptron(
            df=train_df, margin=margin, learning_rate=new_learning_rate, weights=weights, bias=bias
        )
        total_update_counts += update_count

        accuracy = test_accuracy(df=dev_df, weights=weights, bias=bias)
        log.debug(f"  Epoch: {epoch + 1:>2}    Learning rate: {round(new_learning_rate, 5):<8}   Accuracy: {accuracy}")
        print(f"   Epoch: {epoch + 1:>2}    Dev Accuracy: {accuracy}")
        dev_accuracies.append(accuracy)

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch

            best_bias = bias
            best_weights = weights

    log.debug(f"  Best Epoch: {best_epoch + 1:>2}    Dev Accuracy: {best_accuracy}")

    print(
        f"\nc. The total number of updates the learning algorithm (margin {margin} & learning rate {learning_rate}) performs on the training set: {total_update_counts}"
    )
    return best_weights, best_bias, dev_accuracies


if __name__ == "__main__":
    log = init_logger(log_path="./logs", log_file_name=f"{perceptron_type.replace(' ', '_').lower()}.log")

    train_folds, test_folds = prepare_train_test_folds(dfs=dfs)

    # I plan to use same initial setting for CV and online learning.
    initial_weights, initial_bias = initialize_weights_bias(
        rand_start=rand_start, rand_end=rand_end, feature_count=train_df.shape[1] - 1
    )
    log.debug(f"Initial bias: {initial_bias}")
    log.debug(f"Initial weights: {initial_weights}")

    print_answer_1()

    print("2. Majority Baseline")
    _, dev_baseline_accuracy = baseline_accuracy(test_df=test_df, dev_df=dev_df)

    print("\n3. Variants Result")
    log.debug(50 * "-")
    log.debug("Cross Validation Training")

    # Key is the margin and values dictionary of learning rate.
    # For each dictionary of learning rate, key is the learning rate and values list of accuracy.
    # The values in list represent the best cv accuracy of that fold for given margin and learning rate.
    all_cv_accuracies_dict = {}

    for margin in margins:
        if margin not in all_cv_accuracies_dict:
            all_cv_accuracies_dict[margin] = {}

        for learning_rate in learning_rates:
            log.debug(f"Margin: {margin}   Learning rate: {learning_rate}")
            if learning_rate not in all_cv_accuracies_dict[margin]:
                all_cv_accuracies_dict[margin][learning_rate] = []

            for i in range(len(dfs)):
                accuracy = cv_setup(
                    train_fold_df=train_folds[i],
                    test_fold_df=test_folds[i],
                    margin=margin,
                    learning_rate=learning_rate,
                    weights=initial_weights,
                    bias=initial_bias,
                    epochs=10,
                )
                all_cv_accuracies_dict[margin][learning_rate].append(accuracy)
                log.debug(f"  Cross validation (Fold {i}) Accuracy: {accuracy}")

            log.debug("")

    log.debug(f"All CV accuracy dictionary: {all_cv_accuracies_dict}")

    best_lr_parameter = 0
    best_margin_parameter = 0
    best_hyper_parameter = 0
    hyper_parameter_setting = {}

    for margin, cv_lr_accuracies in all_cv_accuracies_dict.items():
        for learning_rate, cv_accuracies in cv_lr_accuracies.items():
            avg_fold_accuracy = 0
            hyper_parameter = f"margin {margin} learning rate {learning_rate}"
            hyper_parameter_setting[hyper_parameter] = {"learning_rate": 0, "margin": 0}

            for accuracy in cv_accuracies:
                avg_fold_accuracy += accuracy

            hyper_parameter_setting[hyper_parameter]["margin"] = margin
            hyper_parameter_setting[hyper_parameter]["learning_rate"] = learning_rate
            hyper_parameter_setting[hyper_parameter]["accuracy"] = avg_fold_accuracy / len(cv_accuracies)

            log.debug(
                f"Average accuracy of all folds with margin {margin} & learning rate {learning_rate} is {avg_fold_accuracy / len(dfs)}"
            )

    best_hyper_parameter = get_key_of_max_value(hyper_parameter_setting)
    best_margin_parameter = hyper_parameter_setting[best_hyper_parameter]["margin"]
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
        margin=best_margin_parameter,
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
