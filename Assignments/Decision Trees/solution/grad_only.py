import math
import pandas as pd
from collections import OrderedDict
from full_trees import get_data_frame_subset, calculate_binary_entropy


attribute_cost = {
    "shape": 10,
    "color": 30,
    "size": 50,
    "material": 100,
}


def get_total_entropy(df):
    label_data = df["label"]
    label_size = label_data.size
    positive_count = list(label_data).count("+")
    negative_count = list(label_data).count("-")

    # print(positive_count, negative_count, label_size)
    p_positive = positive_count / label_size
    p_negative = negative_count / label_size
    # print(p_positive, p_negative)
    return calculate_binary_entropy(pTrue=p_positive, pFalse=p_negative)


def get_info_gain(df, attr, attr_values):
    gain = 0
    total_samples = df.shape[0]
    total_entropy = get_total_entropy(df)
    # print(f"total_entropy: {total_entropy}")
    # print(attr, attr_values)

    for attr_value in attr_values:
        sub_df = get_data_frame_subset(df, attribute=attr, attribute_value=attr_value)
        # print(sub_df)
        samples = sub_df.shape[0]
        positive_count = sub_df["label"].value_counts()["+"] if "+" in sub_df["label"].value_counts() else 0
        negative_count = sub_df["label"].value_counts()["-"] if "-" in sub_df["label"].value_counts() else 0

        p_positive = positive_count / samples
        p_negative = negative_count / samples

        entropy = calculate_binary_entropy(pTrue=p_positive, pFalse=p_negative)
        gain += (samples / total_samples) * entropy
        # print(f"\np_positive: {p_positive}, p_negative: {p_negative}, entropy: {entropy}, gain: {gain}")

    return total_entropy - gain


def calculate_info_gain_T(df):
    info_gain_t = {}
    for attr, cost in attribute_cost.items():
        gain = get_info_gain(df, attr, list(df[attr].unique()))
        info_gain_t[attr] = (gain * gain) / cost

    for attr, gain in info_gain_t.items():
        print(f"Attribute: {attr:<10} GainT: {gain:<25} (Round upto 3 - {round(gain, 3)})")

    return info_gain_t


def calculate_info_gain_S(df):
    info_gain_s = {}
    for attr, cost in attribute_cost.items():
        gain = get_info_gain(df, attr, list(df[attr].unique()))
        info_gain_s[attr] = (pow(2, gain) - 1) / (math.sqrt(cost + 1))

    for attr, gain in info_gain_s.items():
        print(f"Attribute: {attr:<10} GainS: {gain:<25} (Round upto 3 - {round(gain, 3)})")

    return info_gain_s


def get_max_dict_value(dictionary):
    dictionary = OrderedDict(sorted(dictionary.items(), key=lambda x: x[1], reverse=True))
    # print("dictionary", dictionary)
    return next(iter(dictionary))


if __name__ == "__main__":
    df = pd.read_csv("./data/attribute_cost.csv")

    print(
        """1. [8 points] Compute the modified gains GainT and GainS for each attribute using these costs. Fill in your results in the table below. (upto 3 decimal places)"""
    )
    print("GainT (S, A)")
    info_gain_t = calculate_info_gain_T(df)

    print("\nGainS (S, A)")
    info_gain_s = calculate_info_gain_S(df)

    print("\n2. [2 points] For each variant of gain, which feature would you choose as the root?")
    print(f"Best GainT feature for root of the decision tree is '{get_max_dict_value(info_gain_t)}'")
    print(f"Best GainS feature for root of the decision tree is '{get_max_dict_value(info_gain_s)}'\n")
