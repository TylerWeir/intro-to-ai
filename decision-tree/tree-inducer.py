"""
tree-inducer.py

This program takes a tab deliminated file containing voting data
and makes a decision tree to classify voters as Democrat or
Republican. The program then prints out the decision tree and reports
the estimated accuracy of the tree.

Author: Tyler Weir
Date: March 22, 2022
"""

import sys
import math
import random
from os.path import exists

class Node:
    """Represents a Node in the decision tree. Terminal Nodes
    have no children and thus must contain a label."""

    def __init__(self, vote, value, parent=None):
        """Creates a new node."""
        self.parent = parent
        self.children = []
        self.vote = vote
        self.value = value
        self.majority = None

    def add_child(self, child):
        """Adds a child node to the current child list"""
        self.children.append(child)

    def get_children(self):
        """Returns a list containing the children of the node."""
        return self.children

    def get_value(self):
        """Returns the value of the Node."""
        return self.value

    def is_terminal(self):
        """Returns true if the Node is a terminal Node."""
        return len(self.children) == -1

    def get_vote_child(self, vote):
        """Returns the child with the given vote."""
        for child in self.children:
            if child.vote == vote:
                return child

        return None

    def __str__(self):
        return f"(NODE: {self.vote} {self.value})"

class DecisionTree:
    """A Decision Tree classifier."""

    def __init__(self, data):
        """Default constructor for making a new decision tree."""
        self.root = Node("Root", "ISSUE")
        self.build_tree_from_data(data, self.root)

    def build_tree_from_data(self, data, node):
        """Builds the decision tree from the given tab deliminated
        file."""

        ### Decision Tree Algorithm

        # First, Calculate the entropy of the entire training set.
        total_entropy = self.__calc_total_entropy(data)

        # Second, Calculate the information gain of splitting based
        # on each feature
        best_feature = self.__calc_best_feature(data)
        if best_feature == -1:
            # Greatest information gain was 0
            if not self.__has_same_history(data):
                # different voting history
                best_feature = random.randint(0, len(data[0][2])-1)
                node.value = best_feature
            else:
                # same voting history
                # make a leaf node
                node.value = self.__calc_majority(data)
                node.value = self.__find_node_majority(node)
                return
        else:
            node.value = best_feature

        node.majority = self.__calc_majority(data)
        node.majority = self.__find_node_majority(node)

        # Third, Choose the single best feature and divide the data
        # set into two or more discrete groups.

        yes_votes = [(n, l, v) for (n, l, v) in data if v[best_feature] == '+']
        no_votes = [(n, l, v) for (n, l, v) in data if v[best_feature] == '-']
        ab_votes = [(n, l, v) for (n, l, v) in data if v[best_feature] == '.']

        # Fourth, if a subgroup is not uniformly labeled, recurse
        self.__handle_subgroup(yes_votes, node, '+')
        self.__handle_subgroup(no_votes, node, '-')
        self.__handle_subgroup(ab_votes, node, '.')

    def __has_same_history(self, data):
        """Returns true if the entire dataset has the same voting history."""
        history = data[0][2]

        for item in data:
            if item[2] != history:
                return False

        return True

    def __calc_best_feature(self, data):
        """Returns the feature index that results in the greatest information gain.
        if the greatest possible gain is 0 then -1 is returned."""
        best_feature = -1
        greatest_gain = 0

        for i, _ in enumerate(data[0][2]):
            gain = self.__calc_info_gain(data, i)

            if gain > greatest_gain:
                greatest_gain = gain
                best_feature = i

        return best_feature

    def __handle_subgroup(self, subgroup, node, choice):
        """Handles the subgrouping from the decision tree."""

        # IF both Democrat and Republican with same voting record
        # create a leaf that classifies them as the majority

        # If a branch has zero reps classify based on the majority at its parent
        # keep going up the tree until you have a majority

        if len(subgroup) != 0:
            if self.__is_uniform(subgroup):
                # make a leaf node
                label = subgroup[0][1]
                leaf_node = Node(choice, label, parent=node)
                node.add_child(leaf_node)

            else:
                branch_node = Node(choice, 'Some Issue', parent=node)
                node.add_child(branch_node)

                # recurse on the subgroup
                self.build_tree_from_data(subgroup, branch_node)
        else:
            # There are zero reps
            majority = self.__find_node_majority(node)
            leaf_node = Node(choice, majority, parent=node)
            leaf_node.majority = majority
            node.add_child(leaf_node)

    def __calc_total_entropy(self, data):
        """Calculates the entropy of the entire dataset."""
        total = len(data)
        tmp = [party for (name, party, votes) in data]
        num_d = tmp.count('D')
        num_r = tmp.count('R')

        prob_d = num_d/total
        prob_r = num_r/total

        entropy = -prob_d*math.log2(prob_d) -prob_r*math.log2(prob_r)
        return entropy

    def __is_uniform(self, data):
        """Returns true if the data set is uniformly labeled."""
        if len(data) == 0:
            return True

        test = data[0][1]

        for item in data:
            if item[1] != test:
                return False
        return True

    def __find_node_majority(self, node):
        """Recursively search up the tree looking for majority."""

        # First try this node's majority
        if node.majority:
            return node.majority

        # Then try the parent's majority
        if node.parent:
            return self.__find_node_majority(node.parent)
        else:
            # There is no parent
            sys.exit()

    def __calc_majority(self, data):
        """Returns the majority party."""
        D = 0
        R = 0

        for i, rep in enumerate(data):
            if rep[1] == 'R':
                R += 1
            if rep[1] == 'D':
                D += 1

        if R > D:
            return 'R'
        elif D > R:
            return 'D'
        else:
            return None

    def print_tree(self):
        """Prints a text representation of the decision tree."""
        def __print_tree(root, depth):
            """Recursive tree printing function."""

            if depth == 0:
                if type(root.value) == int:
                    print(f'issue {chr(root.value + 65)}:')
                else: 
                    print(f'{root.value}')

                # Recurse on children
                for child in root.children:
                    __print_tree(child, depth + 1)
            else:
                # Print the node
                spaces = ""
                for i in range(depth):
                    spaces += "  "

                if type(root.value) == int:
                    print(f"{spaces}{root.vote} issue {chr(root.value + 65)}:")
                else:
                    print(f"{spaces}{root.vote} {root.value}")

                # Recurse on children
                for child in root.children:
                    __print_tree(child, depth + 1)

        # Enter the recurisve printer
        __print_tree(self.root, 0)

    def __calc_info_gain(self, data, issue):
        """Calculates the information gain of a given decision."""
        total_entropy = self.__calc_total_entropy(data)
        
        # Trim data to just the party and the issue
        data = [(p, v[issue]) for (r, p, v) in data]

        len_data = len(data)
        num_yes = [v for (p, v) in data].count('+')
        num_no = [v for (p, v) in data].count('-')
        num_ab = [v for (p, v) in data].count('.')
        
        entropy_yes  = self.__calc_entropy(data, '+')
        entropy_no = self.__calc_entropy(data, '-')
        entropy_ab = self.__calc_entropy(data, '.')

        gain = total_entropy
        gain -= num_yes/len_data * entropy_yes
        gain -= num_no/len_data * entropy_no
        gain -= num_ab/len_data * entropy_ab
        return gain

    def make_prediction(self, history):
        """Makes a label prediction based on a given voting history."""
        current_node = self.root

        while type(current_node.value) == int:
            issue = current_node.value
            vote = history[issue]
            current_node = current_node.get_vote_child(vote)

        return current_node.value

    def estimate_accuracy(self, data):
        """Estimates the accuracy of the tree by ratio of correct out of total."""
        num_correct = 0
        for item in data:
            prediction = self.make_prediction(item[2])
            if prediction == item[1]:
                num_correct += 1
        return num_correct/len(data)

    def prune(self, node, tuning_data):
        """Prunes the decision tree with a pruning set"""
        # Recurse on children that aren't leafs
        for child in node.children:
            if type(child.value) == int:
                self.prune(child, tuning_data)

        if node.majority == None:
            return

        # Old Accuracy
        old_acc = self.estimate_accuracy(tuning_data)

        # Swap node with leaf classifying as majority
        tmp = node.value
        node.value = self.__find_node_majority(node)

        # New Estimate
        new_acc = self.estimate_accuracy(tuning_data)

        # Lock in the prune if it is better
        if new_acc >= old_acc:
            node.children = []
        else:
            # Otherwise revert
            node.value = tmp


    def __calc_entropy(self, data, choice):
        """Calculates the entropy of asking about choice."""
        num_choice = [v for (p, v) in data].count(choice)
        num_r_choice = data.count(('R', choice))
        num_d_choice = data.count(('D', choice))

        if num_d_choice == 0:
            return 0

        if num_r_choice == 0:
            return 0

        tmp = -(num_r_choice/num_choice) * math.log2(num_r_choice/num_choice)
        tmp += -(num_d_choice/num_choice) * math.log2(num_d_choice/num_choice)
        return tmp

def cross_validation_acc(data):
    """Estimate the accuracy of a tree with leave one out cross
    validation."""
    
    prefix = "Calculating Accuracy:"
    suffix = "Complete"

    printProgressBar(0, len(data), prefix=prefix, suffix=suffix, length=30)
    num_correct = 0

    for i, datum in enumerate(data):
        # make a new dataset without datum
        new_data = data[:]
        new_data.pop(i)

        train_set, tune_set = split_data(new_data)

        # Create the Decision Tree with the training set
        tree = DecisionTree(train_set)

        # Prune the tree with the tuning set
        tree.prune(my_tree.root, tune_set)

        # Make a prediction from datum's features
        prediction = tree.make_prediction(datum[2])
        if prediction == datum[1]:
            num_correct += 1

        # Update progress bar
        printProgressBar(i+1, len(data), prefix=prefix, suffix=suffix, length=30)

    return num_correct/len(data)

"""
This code came from here: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
It was posted by a user named 'Greenstick' 
"""
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '=', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + ' ' * (length - filledLength)
    print(f'{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def split_data(data):
    """Splits a dataset into a training set and a tuning set."""
    # Split the data into a tuning and a training set
    # Every 4th element into the tuning set
    tuning_set = []
    training_set = []
    for i, item in enumerate(data):
        if i%4 == 0:
            tuning_set.append(item)
        else:
            training_set.append(item)

    return training_set[:], tuning_set[:]

if __name__ == '__main__':
    # Checks so silly users don't enter in bad data
    # Check a file was supplied
    if len(sys.argv) != 2:
        print("Usage: python tree-inducer.py [FILE]")
        sys.exit()

    file = sys.argv[1]
    
    # Check that the file exists
    if not exists(file):
        print(f"tree-inducer: cannot access \'{file}\': no such file or directory")
        sys.exit()

    # Read the data set line by line
    data = []
    with open(file) as f:
        data = f.readlines()
    f.close()

    # Split each line by white space
    data = [line.split() for line in data]

    training_set, tuning_set = split_data(data)

    # Create the Decision Tree with the training_set
    my_tree = DecisionTree(training_set)

    # Prune the tree with the tuning_set
    my_tree.prune(my_tree.root, tuning_set)
    print("")
    print("===================")
    print("|| Decision Tree ||")
    print("===================")
    print("")

    # Display the tree
    my_tree.print_tree()
    
    # Cross validation accuracy
    print("")
    acc = cross_validation_acc(data)
    print(f"Accuracy: {acc}")
    print("")

