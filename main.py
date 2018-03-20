import sys
import decision_trees as trees

l = int(sys.argv[1])
k = int(sys.argv[2])

with open(sys.argv[3]) as training_data_file:
    training_data = training_data_file.readlines()

with open(sys.argv[4]) as validation_data_file:
    validation_data = validation_data_file.readlines()

with open(sys.argv[5]) as test_data_file:
    test_data = test_data_file.readlines()

to_print = sys.argv[6]

training_data = [x.strip().split(',') for x in training_data]
validation_data = [x.strip().split(',') for x in validation_data]
test_data = [x.strip().split(',') for x in test_data]

training_set = []
validation_set = []
test_set = []

for i in range(0, len(training_data)):
    my_data = {}
    for j in range(len(training_data[0])):
        my_data[training_data[0][j]] = training_data[i][j]
    training_set.append(my_data)

for i in range(1, len(validation_data)):
    my_data = {}
    for j in range(len(validation_data[0])):
        my_data[validation_data[0][j]] = validation_data[i][j]
    validation_set.append(my_data)

for i in range(1, len(test_data)):
    my_data = {}
    for j in range(len(test_data[0])):
        my_data[test_data[0][j]] = test_data[i][j]
    test_set.append(my_data)

attributes = [attribute for attribute in training_data[0] if attribute != 'Class']

information_gain_tree = trees.Tree(training_set, attributes, 'information gain')
variance_impurity_tree = trees.Tree(training_set, attributes, 'variance impurity')

pruned_information_gain_tree = information_gain_tree.post_prune(l, k, validation_set)
pruned_variance_impurity_tree = variance_impurity_tree.post_prune(l, k, validation_set)

if to_print == 'yes':
    print('\n\n******\nbefore pruning:')
    information_gain_tree.print_tree()
    variance_impurity_tree.print_tree()
    print('\n\n******\nafter pruning:')
    pruned_information_gain_tree.print_tree()
    pruned_variance_impurity_tree.print_tree()


print('\n\n******\nbefore pruning:')
information_gain_tree.print_accuracy(test_set)
variance_impurity_tree.print_accuracy(test_set)

print('\n\n******\nafter pruning:')
pruned_information_gain_tree.print_accuracy(test_set)
pruned_variance_impurity_tree.print_accuracy(test_set)
