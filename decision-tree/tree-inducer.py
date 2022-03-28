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
        
        best_feature = 0
        greatest_gain = -1
       
        for i, _ in enumerate(data[0][2]):
            gain = self.__calc_info_gain(data, i)
            
            print(f'issue {i} has gain {gain}')

            if gain > greatest_gain:
                greatest_gain = gain
                best_feature = i

        print(f'best feature is: {best_feature}')
        
        # Third, Choose the single best feature and divide the data
        # set into two or more discrete groups.

        yes_votes = [(n, l, v) for (n, l, v) in data if v[best_feature] == '+']
        no_votes = [(n, l, v) for (n, l, v) in data if v[best_feature] == '-']
        ab_votes = [(n, l, v) for (n, l, v) in data if v[best_feature] == '.']

        # Fourth, if a subgroup is not uniformly labeled, recurse

        if self.__is_uniform(yes_votes):
            # make a leaf node
            print("yes leaf")
        els
            # recurse on the yes voters
            self.build_tree_from_data(yes_votes)

        if self.__is_uniform(no_votes):
            # make a leaf node
            print("no leaf")
        else: 
            # recurse on the no voters
            self.build_tree_from_data(no_votes)

        if self.__is_uniform(ab_votes):
            # make a leaf node
            print("ab leaf")
        else:
            # recurse on the ab voters
            self.build_tree_from_data(ab_votes)

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

    def calc_accuracy(self):
        """Calculates the estimated accuracy of the decision tree."""
        return None
    
    def __is_uniform(self, data):
        """Returns true if the data set is uniformly labeled."""

        test = data[0][1]

        for item in data:
            if item[0][1] != test:
                return False
        
        return True

    def print_tree(self):
        """Prints a text representation of the decision tree."""
        return None

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

    #TODO only build the tree from some of the data
    # Create the Decision Tree
    my_tree = DecisionTree()
    my_tree.build_tree_from_data(data)

    # Display the tree
    my_tree.print_tree()
