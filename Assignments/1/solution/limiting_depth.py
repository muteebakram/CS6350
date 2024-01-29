import json
import numpy as np
import pandas as pd
from full_trees import get_best_info_gain_attribute, get_data_frame_subset, test_accuracy


df1 = pd.read_csv("./data/CVfolds_new/fold1.csv")
df2 = pd.read_csv("./data/CVfolds_new/fold2.csv")
df3 = pd.read_csv("./data/CVfolds_new/fold3.csv")
df4 = pd.read_csv("./data/CVfolds_new/fold4.csv")
df5 = pd.read_csv("./data/CVfolds_new/fold5.csv")

dfs = [df1, df2, df3, df4, df5]


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
                # print(f"Test  ({i}): ", df_index, df_test.shape)
            else:
                # print("Train indexes: ", df_index, end="\t")
                df_train.append(df_element)

        # print()
        df_train = pd.concat(df_train)
        train_folds.append(df_train)
        # print(f"Train ({i}): ", df_train.shape)

    # print(test_folds)
    # print(train_folds)
    return train_folds, test_folds


def get_majority_label(df):
    edible_count = df["label"].value_counts()["e"] if "e" in df["label"].value_counts() else 0
    poison_count = df["label"].value_counts()["p"] if "p" in df["label"].value_counts() else 0

    # print(f"edible_count: {edible_count}, poison_count: {poison_count}, total: {edible_count + poison_count}")
    if edible_count >= poison_count:  # When equal take gamble and decide to eat :).
        return "e"
    else:
        return "p"


# need to find major label.
def test_majority_label():
    for i in range(len(dfs)):
        print(f"Train fold {i+1} major label {get_majority_label(train_folds[i])}")
        print(f"Test  fold {i+1} major label {get_majority_label(test_folds[i])}")


# need to limit depth
def id3(df, max_depth, tree=None, depth=1):
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
        # print(value, end="\t")
        # Get the dataset with rows set to the attribute value and the attribute column removed.
        sub_df = get_data_frame_subset(df, attribute=best_attribute, attribute_value=value)
        labels = sub_df["label"].unique()
        if len(list(labels)) == 1:
            tree[best_attribute][value]["label"] = list(labels)[0]

        elif depth >= max_depth:
            # print(value, end="\t")
            tree[best_attribute][value]["label"] = get_majority_label(df)

        else:
            sub_tree, sub_tree_depth = id3(sub_df, max_depth, tree=None, depth=depth + 1)
            tree[best_attribute][value] = sub_tree
            current_depth = max(sub_tree_depth, current_depth)

    return tree, current_depth


# There will be 5 train folds with 5224 rows and 5 test folds with 1306 rows.
train_folds, test_folds = prepare_train_test_folds()

print("(c) Cross-validation accuracies for each fold (for limiting depth setting)\n")
best_cv_accuracy = 0
best_decision_tree = None
best_decision_tree_depth = 0

depth_limits = [1, 2, 3, 4, 5, 10, 15]
for depth_limit in depth_limits:
    accuracies = []
    print(f"Depth Limit: {depth_limit}")
    for i in range(len(dfs)):
        tree, depth = id3(train_folds[i], depth_limit)
        # with open(f"./FoldTrees/tree_fold_{i+1}_depth_{depth_limit}.json", "w") as f:
        #     json.dump(tree, f, indent=4)

        accuracy = test_accuracy(test_folds[i], tree)
        print(f"  CV accuracy (Fold {i+1}) : {accuracy}")
        accuracies.append(accuracy)

    std_accuracy = np.std(accuracies)
    avg_accuracy = np.mean(accuracies)
    print(f"  Accuracy             : {avg_accuracy:<25}\n  Standard Deviation   : {std_accuracy:<25}\n")

    if avg_accuracy > best_cv_accuracy:
        best_cv_accuracy = avg_accuracy
        best_decision_tree = tree
        best_decision_tree_depth = depth_limit

print("\n(d) Best depth (for the limiting depth setting)")
print(f"Best depth limit from given limits for the tree is {best_decision_tree_depth} with accuracy {best_cv_accuracy}\n")

train_df = pd.read_csv("./data/train.csv")
print(
    """(e) Accuracy of the trained classifier on the training set (For the limiting depth setting, this would be for the tree with the best depth)"""
)
print(f"Train data accuracy: {test_accuracy(train_df, best_decision_tree)}")

test_df = pd.read_csv("./data/test.csv")
print(
    "\n(f) Accuracy of the trained classifier on the test set (For the limiting depth setting, this would be for the tree with the best depth)"
)
print(f"Test  data accuracy: {test_accuracy(test_df, best_decision_tree)}")
