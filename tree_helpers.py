def classify(tree, datum):
    node = tree.root

    while node.attribute_label not in ['0', '1', '+', '-']:
        if datum.get(node.attribute_label) == '1':
            node = node.positive
        else:
            node = node.negative

    return node.attribute_label in ['1', '+']


def get_tree_as_list(node):
    tree_as_list = [node]

    if node.positive is not None and node.positive.attribute_label not in ['1', '0', '+', '-']:
        tree_as_list += get_tree_as_list(node.positive)
    if node.negative is not None and node.negative.attribute_label not in ['1', '0', '+', '-']:
        tree_as_list += get_tree_as_list(node.negative)

    return tree_as_list


def print_tree(node, depth):
    if node.positive and node.positive.attribute_label not in ['1', '+', '0', '-']:
        print('|' * depth + node.attribute_label + ' = 1 :')
        print_tree(node.positive, depth + 1)
    elif node.positive:
        print('|' * depth + node.attribute_label + ' = 1 : ' + node.positive.attribute_label)

    if node.negative and node.negative.attribute_label not in ['1', '+', '0', '-']:
        print('|' * depth + node.attribute_label + ' = 0 :')
        print_tree(node.negative, depth + 1)
    elif node.negative:
        print('|' * depth + node.attribute_label + ' = 0 : ' + node.negative.attribute_label)


def get_accuracy(tree, data_set):
    classified_data = ['1' if classify(tree, datum) else '0' for datum in data_set]
    pairs = zip(classified_data, [datum.get('Class') for datum in data_set])
    correctly_classified = [(x, y) for (x, y) in pairs if x == y]
    return len(correctly_classified)/len(data_set)
