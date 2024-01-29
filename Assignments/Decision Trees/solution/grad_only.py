import math
import pandas as pd
from full_trees import get_data_frame_subset, get_entropy, get_max_key_by_value


attribute_cost = {
    "shape": 10,
    "color": 30,
    "size": 50,
    "material": 100,
}


def get_info_gain(df, attr, attr_values):
    gain = 0
    total_samples = df.shape[0]
    total_entropy = get_entropy(df, p_value="+", n_value="-")

    for attr_value in attr_values:
        sub_df = get_data_frame_subset(df, attribute=attr, attribute_value=attr_value)
        samples = sub_df.shape[0]

        entropy = get_entropy(sub_df, p_value="+", n_value="-")
        gain += (samples / total_samples) * entropy

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
    print(f"Best GainT feature for root of the decision tree is '{get_max_key_by_value(info_gain_t)}'")
    print(f"Best GainS feature for root of the decision tree is '{get_max_key_by_value(info_gain_s)}'\n")
