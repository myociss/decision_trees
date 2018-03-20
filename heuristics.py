import math


def get_best_attribute(data_subset, attribute_subset, heuristic):
    if heuristic == 'information gain':
        return select_by_information_gain(data_subset, attribute_subset)
    else:
        return select_by_variance_impurity(data_subset, attribute_subset)


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
