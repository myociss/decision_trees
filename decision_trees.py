import tree_helpers
import copy
import random
import heuristics


class Node:
    def __init__(self, data_best_classified_by_this_node, attribute_label):
        self.negative = None
        self.positive = None
        self.data_best_classified_by_this_node = data_best_classified_by_this_node
        self.attribute_label = attribute_label

    def add(self, node):
        if all(datum.get(self.attribute_label) == '1' for datum in node.data_best_classified_by_this_node):
            self.positive = node
        else:
            self.negative = node

    def set_positive(self, node):
        self.positive = node

    def set_negative(self, node):
        self.negative = node


class Tree:
    def __init__(self, complete_training_data, all_attributes, heuristic):
        self.complete_training_data = complete_training_data
        self.all_attributes = all_attributes
        self.heuristic = heuristic
        self.as_list = []
        self.root = self.build_with_id3(self.complete_training_data, self.all_attributes)

    def print_tree(self):
        print('\n\n******\n' + self.heuristic +  ':')
        tree_helpers.print_tree(self.root, 0)

    def print_accuracy(self, data_set):
        print(self.heuristic + ' : ' + str(tree_helpers.get_accuracy(self, data_set)))

    def build_with_id3(self, data_subset, attribute_subset):
        if len(set([datum.get('Class') for datum in data_subset])) == 1:
            return Node(data_subset, set([datum.get('Class') for datum in data_subset]).pop())

        elif len(data_subset) == 2:
            return Node(data_subset, '+')

        elif attribute_subset:

            best_attribute = heuristics.get_best_attribute(data_subset, attribute_subset, self.heuristic)
            new_node = Node(data_subset, best_attribute)
            attributes_copy = [attribute for attribute in attribute_subset if attribute != best_attribute]
            for value in ['0', '1']:
                attribute_has_value = [datum for datum in data_subset if datum.get(best_attribute) == value]
                new_node.add(self.build_with_id3(attribute_has_value, list(attributes_copy)))
            return new_node

        else:
            print("This happens")
            print(data_subset)
            most_common = [datum.get('Class') for datum in data_subset]
            return Node(data_subset, max(set(most_common), key=most_common.count))

    def post_prune(self, l, k, validation_data):
        d_best = copy.deepcopy(self)
        for w in range(1, l):
            d_prime = copy.deepcopy(self)
            m = random.randint(2, k)
            for q in range(1, m):
                d_prime_as_list = tree_helpers.get_tree_as_list(d_prime.root)

                d_prime_length = len(d_prime_as_list)
                p = random.randint(2, d_prime_length - 1)
                node = d_prime_as_list[p]
                most_common = [datum.get('Class') for datum in node.data_best_classified_by_this_node]
                node.attribute_label = max(set(most_common), key=most_common.count)

                node.set_positive(None)
                node.set_negative(None)

            if tree_helpers.get_accuracy(d_prime, validation_data) > tree_helpers.get_accuracy(d_best, validation_data):
                d_best = copy.deepcopy(d_prime)
        return d_best

