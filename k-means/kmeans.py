"""
kmeans.py

This program takes tab deliminated voting data and uses kmeans
to discover natrual clusters within the data.
"""

import sys
import os

def distance_between(history_1, history_2):
    """Calculates the squared distance between to feature space vectors.

    Parameters:
    - history_1: The first feature vector
    - history_2: the other feature vector

    Returns: (float) The squared distance between the feature space vectors.
    """
    if type(history_1[0]) != int:
        # convert to int
        history_1 = convert_to_nums(history_1)
        

    if type(history_2[0]) != int:
        # convert to int
        history_2 = convert_to_nums(history_2)

    # The vectors must be the same length
    assert(len(history_1) == len(history_2))

    dist = 0

    # Add up the squared differences but never sqrt
    for i, _ in enumerate(history_1):
        dist += (history_2 - history_1)**2

    return dist


def convert_to_nums(history):
    """Converts a feature space vector to numerical values.

    Parameters:
    - history: The feature space vector

    Returns: A new vector containing the numerical values
    """
    return [get_num_val(x) for x in history]


def get_num_val(vote):
    """Converts a single vote to a numerical value.

    Parameters:
    - vote (char): The vote to convert.

    Returns: 1 for +, 0 for . and -1 for -.
    """
    if vote == '+':
        return 1
    if vote == '.':
        return 0
    if vote == '-':
        return -1

    print("ERROR! invalid vote value")
    sys.exit()

if __name__ == '__main__':

    # Check that the file exists
    if not os.path.exists(sys.argv[1]):
        print("kmeans: This file does not exist")
        sys.exit()


    # Read the data set line by line
    file = sys.argv[1]
    data = []
    with open(file) as f:
        data = f.readlines()
    f.close()

    # Split each line by white space
    data = [line.split() for line in data]










