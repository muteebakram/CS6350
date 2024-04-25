# %%
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
model_type = "svm_over_trees"

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

# %%
train_df = pd.read_csv("./hw6-data/train.csv")
test_df = pd.read_csv("./hw6-data/test.csv")

tree_train_depth_5_df = []
tree_test_depth_5_df = []

tree_train_depth_10_df = []
tree_test_depth_10_df = []


# %%
def tree_walk(row, tree):
    if "label" in tree:
        # print()
        return tree["label"]

    for key in tree.keys():
        new_key = row[key]
        # print(f"key: {key} -> new_key: {new_key}", end="  ")
        if new_key not in tree[key]:
            # print(f"for key: {key} new_key: {new_key} not in tree.")
            return "NoPath"

        return tree_walk(row, tree[key][new_key])


def test_accuracy(df, tree, store_eval=False):
    df_rows = df.shape[0]
    dict_rows = df.to_dict(orient="records")
    eval_list = []

    if df_rows != len(dict_rows):
        print(f"Error: Mismatch in data frame rows ({df_rows}) and dictionary row of data frames ({len(dict_rows)}).")
        raise ValueError

    correct_prediction = 0
    total_samples = len(dict_rows)

    for index, row in df.iterrows():
        predicted_label = tree_walk(row=dict_rows[index], tree=tree)

        # When there is no path in the Tree take the majority label.
        # Decided to go with this because model needs to predict when it see new examples.
        if predicted_label == "NoPath":
            predicted_label = get_majority_label(df)

        if store_eval:
            eval_list.append(predicted_label)

        if row["label"] == predicted_label:
            correct_prediction += 1

    # print("Accuracy: ", correct_prediction / total_samples)
    return correct_prediction / total_samples, eval_list


def get_majority_label(df, p_label=1, n_label=-1, label_col_name="label"):
    positive_count = df[label_col_name].value_counts()[p_label] if p_label in df[label_col_name].value_counts() else 0
    negative_count = df[label_col_name].value_counts()[n_label] if n_label in df[label_col_name].value_counts() else 0

    if positive_count > negative_count:
        return 1
    else:
        return -1


def get_max_key_by_value(map):
    max_key = ""
    max_val = float("-inf")

    for key, val in map.items():
        if val > max_val:
            max_val = val
            max_key = key

    return max_key


def get_data_frame_subset(df, attribute=None, attribute_value=None):
    if not attribute:
        print(f"Error: No attribute: {attribute} and it's attribute_value: {attribute_value}")
        return None

    df = df[df[attribute] == attribute_value]  # Filter rows with value equal to attribute's value
    df = df.loc[:, df.columns != attribute]  # Remove the attribute column
    return df


def calculate_binary_entropy(pTrue=None, pFalse=None):
    try:
        if pTrue is None or pFalse is None:
            raise AttributeError

        if pTrue == 0.0 or pFalse == 0.0:
            return 0

        return -pTrue * math.log2(pTrue) - pFalse * math.log2(pFalse)

    except Exception:
        print(f"Cannot calculate_binary_entropy for pTrue: {pTrue}, pFalse: {pFalse}")


def get_entropy(df, p_label=1, n_label=-1, label_col_name="label"):
    label_data = df[label_col_name]
    label_size = label_data.size

    # When sub df has no entries return entropy 0 ie no uncertainty.
    if label_size == 0:
        return 0

    # print("label_size", label_size)
    positive_count = df[label_col_name].value_counts()[p_label] if p_label in df[label_col_name].value_counts() else 0
    negative_count = df[label_col_name].value_counts()[n_label] if n_label in df[label_col_name].value_counts() else 0

    # print(f"# of p sample: {positive_count}\n# of n sample: {negative_count}\n# of total samples: {label_size}")
    p_positive = positive_count / label_size
    p_negative = negative_count / label_size
    # print(p_positive, p_negative)

    return calculate_binary_entropy(pTrue=p_positive, pFalse=p_negative)


def get_best_info_gain_attribute(df):
    total_entropy = get_entropy(df, p_label="1", n_label="-1")
    total_samples = df.shape[0]
    attributes = df.columns

    attr_possible_values_dict = {}
    for attr in attributes:
        if attr != "label" and attr not in attr_possible_values_dict:
            attr_possible_values_dict[attr] = list(df[attr].unique())

    information_gain = {}
    for attr, attr_values in attr_possible_values_dict.items():
        if attr not in information_gain:
            information_gain[attr] = 0

        gain = 0
        for attr_value in attr_values:
            sub_df = get_data_frame_subset(df, attribute=attr, attribute_value=attr_value)
            samples = sub_df.shape[0]

            entropy = get_entropy(sub_df, p_label="1", n_label="-1")
            gain += (samples / total_samples) * entropy

        information_gain[attr] += total_entropy - gain

    best_attribute = get_max_key_by_value(information_gain)
    return best_attribute, information_gain[best_attribute]


def id3(df, max_depth, tree=None, depth=1):
    best_attribute, _ = get_best_info_gain_attribute(df)
    best_attribute_possible_values = list(df[best_attribute].unique())
    current_depth = depth
    if not tree:
        tree = {}

    if best_attribute not in tree:
        tree[best_attribute] = {}

    for value in best_attribute_possible_values:
        tree[best_attribute][value] = {}

        # Get the dataset with rows set to the attribute value and the attribute column removed.
        sub_df = get_data_frame_subset(df, attribute=best_attribute, attribute_value=value)
        labels = sub_df["label"].unique()
        if len(list(labels)) == 1:
            tree[best_attribute][value]["label"] = list(labels)[0]

        elif max_depth and depth >= max_depth:
            tree[best_attribute][value]["label"] = get_majority_label(sub_df)

        else:
            # When sub df has only label column then no need split further. To the best_attribute add the majority label.
            if len(sub_df.columns) != 1:
                sub_tree, sub_tree_depth = id3(sub_df, max_depth, tree=None, depth=depth + 1)
                tree[best_attribute][value] = sub_tree
                current_depth = max(sub_tree_depth, current_depth)
            else:
                # print("Best Attribute:", best_attribute, " Value:", value, " Shape:", sub_df.shape)
                tree[best_attribute][value]["label"] = get_majority_label(sub_df)

    return tree, current_depth


# %%
def save_df_as_csv(df, name):
    df.to_csv(f"./hw6-data/{name}.csv", index=False)


def create_tree_df(df, fraction, num_of_trees, tree_max_depth):
    trees = []
    for i in range(num_of_trees):
        sample_df = df.sample(frac=0.1).reset_index(drop=True)
        tree, depth = id3(df=sample_df, max_depth=tree_max_depth)
        trees.append(tree)

    all_rows = []
    dict_rows = df.to_dict(orient="records")
    for index, row in df.iterrows():
        actual_label = row.tolist()[0]
        # example = np.array(row.tolist()[1:])

        tree_predict_row = []
        for i in range(num_of_trees):
            predicted_label = tree_walk(row=dict_rows[index], tree=trees[i])
            tree_predict_row.append(predicted_label)

        tree_predict_row.insert(0, int(actual_label))
        all_rows.append(tree_predict_row)

    new_df = pd.DataFrame(all_rows)
    new_df = new_df.rename(columns={new_df.columns[0]: "label"})

    return new_df


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


def predict_svm(example, weights):
    value = np.dot(weights, example)  # wT x
    return 1 if value > 0 else -1


def test_svm_accuracy(df, weights):
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
        predicted_label = predict_svm(example, weights)
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
    except ZeroDivisionError:
        f1_value = 0

    return avg_accuracy, precision, recall, f1_value


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
            print(f"  Epoch: {epochs - 1}, First Objective: {objectives[0]:<15}, Loss: {total_hinge_loss}")
            continue

        # Mentioned in class to use relative difference.
        cur_objective = objectives[-1]
        prev_objective = objectives[-2]
        difference = np.abs((cur_objective - prev_objective) / prev_objective)
        print(f"  Epoch: {epochs - 1}, Objective difference: {difference:<15}, Loss: {total_hinge_loss}")

        if difference < error:
            break

    return weights, objectives, hinge_losses


def online_setup(train_df, test_df, learning_rate, tradeoff, depth):
    weights, objectives, hinge_losses = svm(df=train_df, lr=learning_rate, tradeoff=tradeoff)
    plot_loss(
        loss=hinge_losses,
        epochs=np.arange(len(hinge_losses)),
        label=f"svm_over_trees_depth_{depth}_lr_{learning_rate}_tradeoff_{tradeoff}",
    )
    avg_accuracy, precision, recall, f1_value = test_svm_accuracy(df=test_df, weights=weights)

    print("")
    print(f"  SVM's Avg_accuracy: {avg_accuracy}")
    print(f"  SVM's Precision: {precision}")
    print(f"  SVM's Recall: {recall}")
    print(f"  SVM's F1-Value: {f1_value}")
    print("")


if __name__ == "__main__":
    # tree_train_depth_5_df = create_tree_df(df=train_df, fraction=0.1, num_of_trees=100, tree_max_depth=5)
    # save_df_as_csv(df=tree_train_depth_5_df, name="tree_depth_5_train")

    # tree_test_depth_5_df = create_tree_df(df=test_df, fraction=0.1, num_of_trees=100, tree_max_depth=5)
    # save_df_as_csv(df=tree_test_depth_5_df, name="tree_depth_5_test")

    # tree_train_depth_10_df = create_tree_df(df=train_df, fraction=0.1, num_of_trees=100, tree_max_depth=10)
    # save_df_as_csv(df=tree_train_depth_10_df, name="tree_depth_10_train")

    # tree_test_depth_10_df = create_tree_df(df=test_df, fraction=0.1, num_of_trees=100, tree_max_depth=10)
    # save_df_as_csv(df=tree_test_depth_10_df, name="tree_depth_10_test")

    # %%

    # %%
    error = 0.00001
    tree_train_depth_5_df = pd.read_csv("./hw6-data/tree_depth_5_train.csv")
    tree_test_depth_5_df = pd.read_csv("./hw6-data/tree_depth_5_test.csv")

    tree_train_depth_10_df = pd.read_csv("./hw6-data/tree_depth_10_train.csv")
    tree_test_depth_10_df = pd.read_csv("./hw6-data/tree_depth_10_test.csv")

    for lr in initial_learning_rates:
        for C in loss_tradeoff_parameter:
            print(f"Depth: 5, Learning rate: {lr}, Tradeoff: {C}")
            online_setup(
                train_df=tree_train_depth_5_df, test_df=tree_test_depth_5_df, learning_rate=lr, tradeoff=C, depth=5
            )

            print(f"Depth: 10, Learning rate: {lr}, Tradeoff: {C}")
            online_setup(
                train_df=tree_train_depth_10_df, test_df=tree_test_depth_10_df, learning_rate=lr, tradeoff=C, depth=10
            )
