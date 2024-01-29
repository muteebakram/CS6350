#!/usr/bin/python3

import math
import json
import pandas as pd


def calculate_binary_entropy(pTrue=None, pFalse=None):
    try:
        if pTrue is None or pFalse is None:
            raise AttributeError

        if pTrue == 0.0 or pFalse == 0.0:
            return 0

        return -pTrue * math.log2(pTrue) - pFalse * math.log2(pFalse)

    except Exception:
        print(f"Cannot calculate_binary_entropy for pTrue: {pTrue}, pFalse: {pFalse}")


def get_entropy(df, p_value="+", n_value="-", label_col_name="label"):
    label_data = df[label_col_name]
    label_size = label_data.size
    positive_count = list(label_data).count(p_value)
    negative_count = list(label_data).count(n_value)

    # print(positive_count, negative_count, label_size)
    p_positive = positive_count / label_size
    p_negative = negative_count / label_size
    # print(p_positive, p_negative)
    return calculate_binary_entropy(pTrue=p_positive, pFalse=p_negative)


def get_data_frame_subset(df, attribute=None, attribute_value=None):
    if not attribute or not attribute_value:
        print(f"Error: No attribute: {attribute} or attribute_value: {attribute_value}")
        return None

    df = df[df[attribute] == attribute_value]  # Filter rows with value equal to attribute's value
    df = df.loc[:, df.columns != attribute]  # Remove the attribute column
    return df


def get_max_key_by_value(map):
    max_key = ""
    max_val = float("-inf")

    for key, val in map.items():
        if val > max_val:
            max_val = val
            max_key = key

    return max_key


def get_best_info_gain_attribute(df):
    total_entropy = get_entropy(df, p_value="e", n_value="p")
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
            # print(attr, attr_value)
            sub_df = get_data_frame_subset(df, attribute=attr, attribute_value=attr_value)
            samples = sub_df.shape[0]

            entropy = get_entropy(sub_df, p_value="e", n_value="p")
            gain += (samples / total_samples) * entropy

        information_gain[attr] = total_entropy - gain

    best_attribute = get_max_key_by_value(information_gain)
    return best_attribute, information_gain[best_attribute]


def id3(df, tree=None, depth=0):
    best_attribute, _ = get_best_info_gain_attribute(df)
    best_attribute_possible_values = list(df[best_attribute].unique())
    # print("best_attribute: ", best_attribute)

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
        else:
            sub_tree, sub_tree_depth = id3(sub_df, tree=None, depth=depth + 1)
            tree[best_attribute][value] = sub_tree
            current_depth = max(sub_tree_depth, current_depth)
            # print("best_attribute: ", best_attribute, " depth: ", current_depth)

    return tree, current_depth


def tree_walk(row, tree):
    if "label" in tree:
        # print()
        return tree["label"]

    for key in tree.keys():
        new_key = row[key]
        # print(f"key: {key} -> new_key: {new_key}", end="  ")
        if new_key not in tree[key]:
            # TODO limiting tree does not have all possible attributes in test folds.
            # print(f"new_key: {new_key} not in tree.")
            return None

        return tree_walk(row, tree[key][new_key])


def test_accuracy(df, tree):
    df_rows = df.shape[0]
    dict_rows = df.to_dict(orient="records")

    if df_rows != len(dict_rows):
        print(f"Error: Mismatch in data frame rows ({df_rows}) and dictionary row of data frames ({len(dict_rows)}).")
        raise ValueError

    correct_prediction = 0
    total_samples = len(dict_rows)
    # print("Total Samples: ", total_samples)

    for index, row in df.iterrows():
        predicted_label = tree_walk(row=dict_rows[index], tree=tree)
        if predicted_label is None:
            # print("Failed to predict for row: ", row)
            continue

        if row["label"] == predicted_label:
            correct_prediction += 1

    # print("Accuracy: ", correct_prediction / total_samples)
    return correct_prediction / total_samples


if __name__ == "__main__":
    df = pd.read_csv("./data/train.csv")
    root_feature, info_gain = get_best_info_gain_attribute(df)

    print("(a) [2 points] The root feature that is selected by your algorithm")
    print(root_feature)

    print("\n(b) [2 point] Information gain for the root feature")
    print(info_gain)

    tree, depth = id3(df)
    with open("tree.json", "w") as f:
        json.dump(tree, f, indent=4)

    print("\n(c) [2 points] Maximum depth of the tree that your implementation gives")
    print(depth + 1)  # Need to count label for depth

    print("\n(d) [3 points] Accuracy on the training set")
    print(test_accuracy(df, tree))

    print("\n(e) [5 points] Accuracy on the test set")
    df = pd.read_csv("./data/test.csv")
    tree, depth = id3(df)
    print(test_accuracy(df, tree))

    print("\n(a) Entropy of the data")
    print(get_entropy(df, p_value="e", n_value="p"))

    print("\n(b) Best feature and its information gain")
    print(f"{root_feature}: {info_gain}")
