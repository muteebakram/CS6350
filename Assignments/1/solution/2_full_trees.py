#!/usr/bin/python3

import math
from data import Data
from collections import OrderedDict


class Node:
    def __init__(self):
        self.data = None
        self.array_nodes = []


class ID3:
    def __init__(self, data_obj):
        self.data_obj = data_obj
        self.total_entropy = self.get_total_entropy()
        self.attribute_possible_values = self.get_attribute_possible_values()
        self.attribute_data = self.prepare_attribute_data()
        self.information_gain = self.calculate_info_gain()
        self.root_node = self.id3_root()
        self.decision_tree = self.build_tree()

    def get_total_entropy(self):
        label_data = self.data_obj.get_column("label")
        label_size = label_data.size
        edible_count = list(label_data).count("e")
        poison_count = list(label_data).count("p")

        # print(edible_count, poison_count, label_size)
        p_edible = edible_count / label_size
        p_poison = poison_count / label_size
        # print(p_edible, p_poison)
        return calculate_binary_entropy(pTrue=p_edible, pFalse=p_poison)

    def get_attribute_subset(self, attribute_name, attribute_value):
        sub_dataset = self.data_obj.get_row_subset(attribute_name=attribute_name, attribute_value=attribute_value)

        # print(sub_dataset.raw_data)
        label_sub_dataset = list(sub_dataset.raw_data[:, 0])
        # print(label_sub_dataset)
        total_count = len(label_sub_dataset)
        edible_count = list(label_sub_dataset).count("e")
        poison_count = list(label_sub_dataset).count("p")

        # print(edible_count, poison_count, total_count)

        return {
            "attribute_name": attribute_name,
            "attribute_value": attribute_value,
            "edible_count": edible_count,
            "poison_count": poison_count,
            "total_count": total_count,
        }

        # print(self.data_obj.get_column("cap-shape", sub))
        # print('\n'.join(''.join(str(cell) for cell in row) for row in sub.raw_data))

    def prepare_attribute_data(self):
        attribute_meta_data = {}
        for attribute, attribute_values in self.attribute_possible_values.items():
            for attribute_value in attribute_values:
                if attribute not in attribute_meta_data:
                    attribute_meta_data[attribute] = []

                attribute_meta_data[attribute].append(
                    self.get_attribute_subset(attribute_name=attribute, attribute_value=attribute_value)
                )
                # print(f"attribute: {attribute} -> attribute_value: {attribute_value}", end="\t")

            # print()

        # print(attribute_meta_data)
        for attribute, attribute_datas in attribute_meta_data.items():
            for attribute_data in attribute_datas:
                attribute_value = attribute_data["attribute_value"]
                edible_count = attribute_data["edible_count"]
                poison_count = attribute_data["poison_count"]
                total_count = attribute_data["total_count"]

                if total_count == 0:
                    attribute_data["sub_entropy"] = 0
                    continue

                p_edible = edible_count / total_count
                p_poison = poison_count / total_count

                attribute_data["sub_entropy"] = calculate_binary_entropy(pTrue=p_edible, pFalse=p_poison)
                # print(
                #     f"attribute: {attribute}, attribute_value: {attribute_value}, p_edible: {p_edible}, p_poison: {p_poison}, \
                #     entropy: {attribute_data['sub_entropy']}, edible_count: {edible_count}, poison_count: {poison_count}, total_count: {total_count}"
                # )

        # print(attribute_meta_data)
        return attribute_meta_data

    def get_attribute_possible_values(self):
        attribute_possible_values_dict = {}
        for attribute in self.data_obj.index_column_dict.values():
            if attribute == "label":
                continue

            # print(attribute, self.data_obj.attributes[attribute].possible_vals)
            attribute_possible_values_dict[attribute] = list(self.data_obj.attributes[attribute].possible_vals)

        return attribute_possible_values_dict

    def calculate_info_gain(self):
        total_samples = self.data_obj.__len__()
        # print("total_samples: ", total_samples)

        info_gain = {}  # Key is attribute_label and value is info gain for the attribute
        for attribute, attribute_values in self.attribute_data.items():
            gain = 0
            for attribute_value in attribute_values:
                total_count = attribute_value["total_count"]
                sub_entropy = attribute_value["sub_entropy"]
                gain += (total_count / total_samples) * sub_entropy

            # print(f"attribute: {attribute}, total_entropy: {self.total_entropy}, gain: {gain}")
            info_gain[attribute] = self.total_entropy - gain

        # print(info_gain)
        info_gain = OrderedDict(sorted(info_gain.items(), key=lambda x: x[1], reverse=True))
        return info_gain

    def id3_root(self):
        # for key, value in self.information_gain.items():
        #     print(key, value)
        data = next(iter(self.information_gain))
        # print(root)

        root = Node()
        root.data = data
        return root

    def build_tree(self):
        pass
        # get max info gain
        # if not tree:
        #     tree[max_info_gain]

        # get data subset - with data of all equal to attr value and remove attr column
        # check subset with label if only one -> add
        # id3(subset, attribute - a)


def calculate_binary_entropy(pTrue=None, pFalse=None):
    try:
        if pTrue is None or pFalse is None:
            raise AttributeError

        if pTrue == 0.0 or pFalse == 0.0:
            return 0

        return -pTrue * math.log2(pTrue) - pFalse * math.log2(pFalse)

    except Exception:
        print(f"Cannot calculate_binary_entropy for pTrue: {pTrue}, pFalse: {pFalse}")


def traverse(tree):
    if not tree:
        return

    print(tree.data)
    for node in tree.array_nodes:
        # print(node)
        if not node:
            continue

        for edge, next_node in node.items():
            print(edge, end="\t")
            traverse(next_node)
        print()


def test_tree_traversal():
    tree = Node()
    tree.data = "Hi"

    left = Node()
    left.data = "World"
    left.array_nodes = []

    right = Node()
    right.data = "Muteeb"
    right.array_nodes = [None]

    tree.array_nodes.append({"left": left})
    tree.array_nodes.append({"right": right})

    traverse(tree)


train_data_obj = Data(fpath="./data/train.csv")
test_data_obj = Data(fpath="./data/test.csv")

id3 = ID3(train_data_obj)
id3.id3_root()
# print(id3.data_obj.index_column_dict)
# print()
# print(id3.data_obj.column_index_dict)
# print(id3.data_obj.get_attribute_possible_vals("veil-type"))
# print(id3.data_obj.get_attribute_possible_vals("stalk-root"))
# print(id3.get_attribute_subset("cap-shape", "x"))
test_tree_traversal()
