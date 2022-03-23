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

    def build_tree_from_data(self, filename):
        """Builds the decision tree from the given tab deliminated
        file."""

        ### Decision Tree Algorithm

        # First, Calculate the entropy of the entire training set.

        # Second, Calculate the information gain of splitting based
        # on each feature

        # Third, Choose the single best feature and divide the data
        # set into two or more discrete groups.

        # Fourth, if a subgroup is not uniformly labeled, recurse



    def calc_accuracy(self):
        """Calculates the estimated accuracy of the decision tree."""
        return None

    def print_tree(self):
        """Prints a text representation of the decision tree."""
        return None

    def __calc_entropy(self):
        """Calculates the entropy of a given decision."""
        return None

if __name__ == '__main__':

    # Check if the file exists and is the correct format
    file = sys.argv[1]


