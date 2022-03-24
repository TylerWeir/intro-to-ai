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

class DecisionTree:
    """A Decision Tree classifier."""

    def __init__(self):
        """Default constructor for making a new decision tree."""
        return None

    def build_tree_from_data(self, data):
        """Builds the decision tree from the given tab deliminated
        file."""

        ### Decision Tree Algorithm

        # First, Calculate the entropy of the entire training set.
        total_entropy = self.__calc_total_entropy(data)

        # Second, Calculate the information gain of splitting based
        # on each feature
        
        #TODO only first question right now
        print("calc info gain from first feature")
        gain = self.__calc_info_gain(data, 0)

        # Third, Choose the single best feature and divide the data
        # set into two or more discrete groups.

        # Fourth, if a subgroup is not uniformly labeled, recurse

    def __calc_total_entropy(self, data):
        """Calculates the entropy of the entire dataset."""
        total = len(data)
        tmp = [party for (name, party, votes) in data]
        num_d = tmp.count('D')
        num_r = total-num_d

        prob_d = num_d/total
        prob_r = num_r/total

        entropy = -prob_d*math.log2(prob_d) -prob_r*math.log2(prob_r)
        return entropy


    def calc_accuracy(self):
        """Calculates the estimated accuracy of the decision tree."""
        return None

    def print_tree(self):
        """Prints a text representation of the decision tree."""
        return None

    def __calc_info_gain(self, data, issue):
        """Calculates the information gain of a given decision."""
        total_entropy = self.__calc_total_entropy(data)
        
        # Trim data to just the party and the issue
        data = [(p, v[issue]) for (r, p, v) in data]

        num_r_yes = data.count(('R', '+'))
        num_yes = [v for (p, v) in data].count('+')
        
        entropy_yes  = self.__calc_entropy(data, '+')
        entropy_other = self.__calc_entropy(data, '-.')

    def __calc_entropy(self, data, choice):
        """Calculates the entropy of asking about choice."""
        num_yes = [v for (p, v) in data].count('+')
        num_r_yes = data.count(('R', '+'))
        num_d_yes = data.count(('D', '+'))

        if choice == '+':
            tmp = -(num_r_yes/num_yes) * math.log2(num_r_yes/num_yes)
            tmp += -(num_d_yes/num_yes) * math.log2(num_d_yes/num_yes)
            return tmp

        # Otherwise calculating the entropy of no and abstain
        num_r = [p for (p, v) in data].count('R')
        num_r_no = num_r - num_r_yes
        num_no = len(data) - num_yes
        num_d_no = num_r - num_r_no

        tmp = -(num_r_no/num_no) * math.log2(num_r_no/num_no)
        tmp += -(num_d_no/num_no) * math.log2(num_d_no/num_no)
        return tmp

    class Node:
        """Represents a Node in the decision tree. Terminal Nodes
        have no children and thus must contain a label."""

        def __init__(self, value, parent=None, children=[]):
            """Creates a new node."""
            self.parent = parent
            self.children = children

            self.value = value

        def get_children(self):
            """Returns a list containing the children of the node."""
            return self.children

        def get_value(self):
            """Returns the value of the Node."""
            return self.value

        def is_terminal(self):
            """Returns true if the Node is a terminal Node."""
            return len(self.children) == 0

if __name__ == '__main__':

    # TODO Checks so silly users don't enter in bad data
    # Check if the file exists and is the correct format
    file = sys.argv[1]

    # Read the data set line by line
    data = []
    with open(file) as f:
        data = f.readlines()
    f.close()

    # Split each line by white space
    data = [line.split() for line in data]
    print("finished reading the file")

    #TODO only build the tree from some of the data
    # Create the Decision Tree
    my_tree = DecisionTree()
    my_tree.build_tree_from_data(data)

    # Display the tree
    my_tree.print_tree()
