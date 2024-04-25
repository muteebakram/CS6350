import numpy as np
import pandas as pd
from common import init_logger, add_bias_to_df, prepare_train_test_folds, get_key_of_max_value, test_accuracy, plot_loss

model_type = "svm"

error = 0.1
# error = 0.001
# error = 0.0001

random_seed = 42
np.random.seed(random_seed)

# initial_learning_rates = [1.0]
# loss_tradeoff_parameter = [100.0]
initial_learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]
loss_tradeoff_parameter = [
    10000.0,
    1000.0,
    100.0,
    10.0,
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


def svm(df, lr, tradeoff):
    objectives = []
    hinge_losses = []

    epochs = 1
    weights = np.zeros(df.shape[1] - 1)  # features + bias

    while True:
        # Reset hinge loss for every epoch
        total_objective = 0
        total_hinge_loss = 0

        # Change learning rate for every epoch
        lr_t = lr / (1 + epochs)

        # Shuffle dataset and for each example update weights.
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        for _, row in df.iterrows():
            label = row.tolist()[0]
            example = np.array(row.tolist()[1:])

            if label * (np.dot(weights, example)) <= 1:
                weights = ((1 - lr_t) * weights) + (lr_t * tradeoff * label * example)
            else:
                weights = (1 - lr_t) * weights

        # Updates done increment epoch.
        epochs += 1

        for _, row in df.iterrows():
            label = row.tolist()[0]
            example = np.array(row.tolist()[1:])

            hinge_loss = max(0, 1 - (label * np.dot(weights, example)))
            new_objective = 0.5 * np.dot(weights, weights) + tradeoff * hinge_loss

            total_objective += new_objective
            total_hinge_loss += hinge_loss

        objectives.append(total_objective)
        hinge_losses.append(total_hinge_loss)

        # Only new objective is present.
        if len(objectives) == 1:
            log.debug(f"  Epoch: {epochs - 1}, First Objective: {objectives[0]:<15}, Loss: {total_hinge_loss}")
            continue

        # Mentioned in class to use relative difference.
        cur_objective = objectives[-1]
        prev_objective = objectives[-2]
        difference = np.abs((cur_objective - prev_objective) / prev_objective)
        log.debug(f"  Epoch: {epochs - 1}, Objective difference: {difference:<15}, Loss: {total_hinge_loss}")

        if difference < error:
            break

    return weights, objectives, hinge_losses


def online_setup(train_df, test_df, learning_rate, tradeoff):
    weights, objectives, hinge_losses = svm(df=train_df, lr=learning_rate, tradeoff=tradeoff)
    plot_loss(loss=hinge_losses, epochs=np.arange(len(hinge_losses)), label=model_type.upper())
    avg_accuracy, precision, recall, f1_value = test_accuracy(df=test_df, weights=weights)

    log.debug("")
    log.debug(f"SVM's Avg_accuracy: {avg_accuracy}")
    log.debug(f"SVM's Precision: {precision}")
    log.debug(f"SVM's Recall: {recall}")
    log.debug(f"SVM's F1-Value: {f1_value}")


def run_cv(train_fold_df, test_fold_df, learning_rate, tradeoff):
    weights, _, _ = svm(df=train_fold_df, lr=learning_rate, tradeoff=tradeoff)
    avg_accuracy, precision, recall, f1_value = test_accuracy(df=test_fold_df, weights=weights)

    log.debug(f"  Avg_accuracy: {avg_accuracy}, Precision: {precision}, Recall: {recall}, F1-Value: {f1_value}")
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
        log.debug(f"Hyper parameter: {key:<15} Best CV: {cv_fold}   F1-Value: {best_f1_value}")

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

    # best_lr_param, best_tradeoff_param = [0.001, 1000.0]
    train_df = add_bias_to_df(train_df)
    test_df = add_bias_to_df(test_df)

    error = 0.0001  # Run for more epochs with best parameters.
    log.debug("")
    log.debug(f"Online learning with epoch error: {error}")
    online_setup(train_df=train_df, test_df=test_df, learning_rate=best_lr_param, tradeoff=best_tradeoff_param)
