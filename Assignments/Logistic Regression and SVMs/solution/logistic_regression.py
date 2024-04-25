import numpy as np
import pandas as pd
from common import (
    init_logger,
    initialize_weights,
    add_bias_to_df,
    prepare_train_test_folds,
    get_key_of_max_value,
    test_accuracy,
    plot_loss,
)

model_type = "logistic_regression"

# error = 0.1
# error = 0.0005
error = 0.0001  # Run for more epochs.
# error = 0.0001

random_seed = 42
np.random.seed(random_seed)


initial_learning_rates = [1.0]
loss_tradeoff_parameter = [0.10]
initial_learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]
loss_tradeoff_parameter = [
    1000.0,
    100.0,
    10.0,
    1,
    0.1,
    0.01,
    0.001,
    0.0001,
]  # Mentioned in class to add higher tradeoff parameters.

df0 = pd.read_csv("./hw6-data/CVSplits/training00.csv")
df1 = pd.read_csv("./hw6-data/CVSplits/training01.csv")
df2 = pd.read_csv("./hw6-data/CVSplits/training02.csv")
df3 = pd.read_csv("./hw6-data/CVSplits/training03.csv")
df4 = pd.read_csv("./hw6-data/CVSplits/training04.csv")

dfs = [df0, df1, df2, df3, df4]

train_df = pd.read_csv("./hw6-data/train.csv")
test_df = pd.read_csv("./hw6-data/test.csv")


# Piazza post @337
def sigmoid(u):
    if u <= -10:
        return 0
    elif u >= 10:
        return 1

    return 1 / (1 + np.exp(u))


def logistic_regression(df, lr, tradeoff):
    prev_objective = -1
    logistic_losses = []

    epochs = 1
    weights = np.array(initialize_weights(-pow(10, -3), pow(10, -3), df.shape[1] - 1))

    while True:
        # Shuffle dataset and for each example update weights.
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # Pick an example from the shuffled dataset
        random_row = df.sample(n=1).values.tolist()[0]
        label = random_row[0]
        example = np.array(random_row[1:])

        u = label * np.dot(weights, example)
        # log.debug(f"label: {label}, u : {u}, sigmoid(u): {sigmoid(u)}")
        if u <= -10:
            u = -10
        elif u >= 10:
            u = 10

        # Gradient decent
        gradient = (1 - sigmoid(u)) * (-label * example) + ((2 / tradeoff) * weights)
        # Weights Update
        weights = weights - lr * gradient

        logistic_loss = np.log(1 + np.exp(-u))
        new_objective = np.log(1 + np.exp(-u)) + (np.dot(weights, weights) * (1 / tradeoff))

        # log.debug(f" logistic_loss : {logistic_loss}, new_objective: {new_objective}")

        # Updates done increment epoch.
        epochs += 1

        if new_objective < -pow(10, 6):
            new_objective = -pow(10, 6)
        elif new_objective > pow(10, 6):
            new_objective = pow(10, 6)

        if prev_objective == -1:
            prev_objective = new_objective
            log.debug(f"  Epoch: {epochs - 1}, First Objective: {prev_objective:<15} Loss: {logistic_loss}")
            continue

        # Mentioned in class to use relative difference.
        difference = np.abs((new_objective - prev_objective) / prev_objective)
        log.debug(f"  Epoch: {epochs - 1}, Objective difference: {difference:<15} Loss: {logistic_loss}")
        if (difference <= error) or epochs > 20:
            break

        logistic_losses.append(logistic_loss)

        prev_objective = new_objective

    return weights, logistic_losses


def online_setup(train_df, test_df, learning_rate, tradeoff):
    weights, logistic_losses = logistic_regression(df=train_df, lr=learning_rate, tradeoff=tradeoff)
    plot_loss(loss=logistic_losses, epochs=np.arange(len(logistic_losses)), label=model_type)
    avg_accuracy, precision, recall, f1_value = test_accuracy(df=test_df, weights=weights)

    log.debug("")
    log.debug(f"  Logistic Regression's Avg_accuracy: {avg_accuracy}")
    log.debug(f"  Logistic Regression's Precision: {precision}")
    log.debug(f"  Logistic Regression's Recall: {recall}")
    log.debug(f"  Logistic Regression's F1-Value: {f1_value}")


def run_cv(train_fold_df, test_fold_df, learning_rate, tradeoff):
    weights, logistic_losses = logistic_regression(df=train_fold_df, lr=learning_rate, tradeoff=tradeoff)
    avg_accuracy, precision, recall, f1_value = test_accuracy(df=test_fold_df, weights=weights)

    log.debug(f"CV Avg_accuracy: {avg_accuracy}, Precision: {precision}, Recall: {recall}, F1-Value: {f1_value}")
    return f1_value


def cv_setup(train_folds, test_folds):
    all_cv_f1_values_dict = {}

    for lr in initial_learning_rates:
        for tradeoff in loss_tradeoff_parameter:
            key = f"{lr}:{tradeoff}"
            if key not in all_cv_f1_values_dict:
                all_cv_f1_values_dict[key] = []

            for i in range(len(dfs)):
                log.debug("")
                log.debug(f"CV (Fold {i + 1}) with lr: {lr} & tradeoff: {tradeoff}")
                f1_value = run_cv(
                    train_fold_df=train_folds[i],
                    test_fold_df=test_folds[i],
                    learning_rate=lr,
                    tradeoff=tradeoff,
                )
                all_cv_f1_values_dict[key].append(f1_value)

    log.debug("")
    for key, f1_values in all_cv_f1_values_dict.items():
        best_f1_value = max(f1_values)
        cv_fold = f1_values.index(best_f1_value)
        all_cv_f1_values_dict[key] = max(f1_values)  # Change value from list to a single number.
        log.debug(f"Hyper parameter: {key:<15}, Best CV: {cv_fold}, F1-Value: {best_f1_value}")

    log.debug("")
    log.debug(f"All CV Best F1 Dict: {all_cv_f1_values_dict}")

    hyper_param_key = get_key_of_max_value(all_cv_f1_values_dict)
    best_lr_param, best_tradeoff_param = hyper_param_key.split(":")

    return float(best_lr_param), float(best_tradeoff_param)


if __name__ == "__main__":
    log = init_logger(log_path="./logs", log_file_name=f"{model_type}.log")
    log.debug(f"{model_type.upper()} Epoch Error: {error}, Random Seed: {random_seed}")
    log.debug(f"{model_type.upper()} Learning rates: {initial_learning_rates}, Tradeoff: {loss_tradeoff_parameter}")

    for df in dfs:
        df = add_bias_to_df(df)

    train_folds, test_folds = prepare_train_test_folds(dfs=dfs)
    best_lr_param, best_tradeoff_param = cv_setup(train_folds, test_folds)
    log.debug(f"Best Learning Rate: {best_lr_param} and Tradeoff: {best_tradeoff_param}")
    print(f"Best Learning Rate: {best_lr_param} and Tradeoff: {best_tradeoff_param}")

    best_lr_param, best_tradeoff_param = [0.0001, 1]

    log.debug("")
    log.debug(f"Online learning with epoch error: {error}")
    train_df = add_bias_to_df(train_df)
    test_df = add_bias_to_df(test_df)
    online_setup(train_df=train_df, test_df=test_df, learning_rate=best_lr_param, tradeoff=best_tradeoff_param)
