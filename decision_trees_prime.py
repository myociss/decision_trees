import sys
import math
import copy
import random


def select_by_information_gain(data_subset, attribute_subset):
    all_attribute_information_gains = {attr: information_gain(data_subset, attr) for attr in attribute_subset}
    return max(all_attribute_information_gains, key=all_attribute_information_gains.get)


def select_by_variance_impurity(data_subset, attribute_subset):
    all_attribute_information_gains = {attr: impurity_gain(data_subset, attr) for attr in attribute_subset}
    return max(all_attribute_information_gains, key=all_attribute_information_gains.get)


def information_gain(data_subset, attr):

    p_0 = [x.get(attr) for x in data_subset].count('0') / len(data_subset)
    p_1 = 1 - p_0

    zeros = [x for x in data_subset if x.get(attr) is '0']
    ones = [x for x in data_subset if x.get(attr) is '1']

    return entropy(data_subset) - p_0 * entropy(zeros) - p_1 * entropy(ones)


def impurity_gain(data_subset, attr):
    p_0 = [x.get(attr) for x in data_subset].count('0') / len(data_subset)
    p_1 = 1 - p_0

    zeros = [x for x in data_subset if x.get(attr) is '0']
    ones = [x for x in data_subset if x.get(attr) is '1']
    return variance_impurity(data_subset) - (p_0 * variance_impurity(zeros) + p_1 * variance_impurity(ones))


def entropy(data_subset):
    if len(data_subset) == 0:
        return 0.0
    else:
        p_pos = [x.get('Class') for x in data_subset].count('1') / len(data_subset)
        p_neg = 1 - p_pos
        if p_pos == 0.0 or p_neg == 0.0:
            return 0.0
        else:
            return -p_pos * math.log(p_pos, 2) - p_neg * math.log(p_neg, 2)


def variance_impurity(data_subset):
    if len(data_subset) == 0:
        return 0.0
    else:
        percent_positive_data = [x.get('Class') for x in data_subset].count('1') / len(data_subset)
        percent_negative_data = 1 - percent_positive_data
        return percent_positive_data * percent_negative_data


def id3(data_subset, attribute_subset, heuristic):
    if all(datum.get('Class') == '1' for datum in data_subset):
        return Node(data_subset, '+')
    elif all(datum.get('Class') == '0' for datum in data_subset):
        return Node(data_subset, '-')
    elif not attribute_subset:
        print("This never happens")
        most_common = [datum.get('Class') for datum in data_subset]
        return Node(data_subset, max(set(most_common), key=most_common.count))
    else:
        if heuristic == 'information gain':
            best_attribute = select_by_information_gain(data_subset, attribute_subset)
        else:
            best_attribute = select_by_variance_impurity(data_subset, attribute_subset)

        has_best_attribute_true = [datum for datum in data_subset if datum.get(best_attribute) == '1']
        has_best_attribute_false = [datum for datum in data_subset if datum.get(best_attribute) == '0']
        new_node = Node(data_subset, best_attribute)
        attributes_copy = [attribute for attribute in attribute_subset if attribute != best_attribute]
        new_node.negative = id3(has_best_attribute_false, attributes_copy, heuristic)
        new_node.positive = id3(has_best_attribute_true, list(attributes_copy), heuristic)
        return new_node


class Node:
    def __init__(self, data_best_classified_by_this_node, attribute_label):
        self.negative = None
        self.positive = None
        self.data_best_classified_by_this_node = data_best_classified_by_this_node
        self.attribute_label = attribute_label


class Tree:
    def __init__(self, complete_data_set, v_s, all_attributes, heuristic):
        self.complete_data_set = complete_data_set
        self.all_attributes = all_attributes
        self.heuristic = heuristic
        self.v_s = v_s
        self.as_list = []
        self.root = id3(self.complete_data_set, self.all_attributes, heuristic)

    def classify(self, datum):

        node = self.root
        #print(node.attribute_label)
        while node.attribute_label not in ['0', '1', '+', '-']:

            if datum.get(node.attribute_label) == '1':
                if not node.positive:
                    print('positive is none')
                    print(node.attribute_label)
                node = node.positive
            else:
                if not node.negative:
                    print('negative is none')
                    print(node.attribute_label)
                node = node.negative
        return node.attribute_label in ['1', '+']

    def post_prune(self, l, k):
        d_best = copy.deepcopy(self)
        d_best.get_as_list(d_best.root)
        for w in range(1, l):
            d_prime = copy.deepcopy(self)
            m = random.randint(2, k)
            for q in range(1, m):
                d_prime.get_as_list(d_prime.root)

                d_prime_length = len(d_prime.as_list)
                p = random.randint(2, d_prime_length - 1)
                my_list = d_prime.as_list
                node = my_list[p]
                most_common = [datum.get('Class') for datum in node.data_best_classified_by_this_node]
                node.attribute_label = max(set(most_common), key=most_common.count)

                node.positive = None
                node.negative = None

            if d_prime.get_accuracy() > d_best.get_accuracy():
                d_best = copy.deepcopy(d_prime)

        return d_best

    def get_as_list(self, node):
        if node == self.root:
            self.as_list = []
        self.as_list.append(node)
        if node.positive is not None and node.positive.attribute_label not in ['1', '0', '+', '-']:
            self.get_as_list(node.positive)
        if node.negative is not None and node.negative.attribute_label not in ['1', '0', '+', '-']:
            self.get_as_list(node.negative)

    def print_tree(self, node, depth):
        if node.positive and node.positive.attribute_label not in ['1', '+', '0', '-']:
            print('|' * depth + node.attribute_label + ' = 1 :')
            self.print_tree(node.positive, depth + 1)
        elif node.positive:
            print('|' * depth + node.attribute_label + ' = 1 : ' + node.positive.attribute_label)

        if node.negative and node.negative.attribute_label not in ['1', '+', '0', '-']:
            print('|' * depth + node.attribute_label + ' = 0 :')
            self.print_tree(node.negative, depth + 1)
        elif node.negative:
            print('|' * depth + node.attribute_label + ' = 0 : ' + node.negative.attribute_label)

    def get_accuracy(self):
        classified_data = ['1' if self.classify(datum) else '0' for datum in self.v_s]

        pairs = zip(classified_data, [datum.get('Class') for datum in self.v_s])
        correctly_classified = [(x, y) for (x,y) in pairs if x == y]
        return len(correctly_classified)/len(self.v_s)


with open(sys.argv[1]) as f:
    content = f.readlines()

with open(sys.argv[2]) as f1:
    validation_data = f1.readlines()

content = [x.strip().split(',') for x in content]
validation_data = [x.strip().split(',') for x in validation_data]

training_set = []
validation_set = []

for i in range(0, len(content)):
    my_data = {}
    for j in range(len(content[0])):
        my_data[content[0][j]] = content[i][j]
    training_set.append(my_data)

for i in range(1, len(validation_data)):
    my_data = {}
    for j in range(len(validation_data[0])):
        my_data[validation_data[0][j]] = validation_data[i][j]
    validation_set.append(my_data)

content[0].remove('Class')
tree = Tree(training_set, validation_set, content[0], 'information gain')

tree.print_tree(tree.root, 0)
new_tree = tree.post_prune(30, 9)



print("\n\n")
#new_tree.print_tree(new_tree.root, 0)
print(tree.get_accuracy())
print(new_tree.get_accuracy())
