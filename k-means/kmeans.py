"""
kmeans.py

This program takes tab deliminated voting data and uses kmeans
to discover natrual clusters within the data.

Author: Tyler Weir
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
        dist += (history_2[i] - history_1[i])**2

    return dist


def convert_to_nums(history):
    """Converts a feature space vector to numerical values.

    Parameters:
    - history: The feature space vector

    Returns: A new vector containing the numerical values
    """
    return [get_num_val(x) for x in history]


def get_furthest_apart(data):
    """Returns the two representatives who are the furthest apart
    from eachother in the data set.

    Parameters:
    - data: The voting data read in

    Returns: The indeces representatives furthest apart as ints"""

    furthest = (0,0)
    furthest_dist = 0

    for i, rep_a in enumerate(data):
        for j, rep_b in enumerate(data):

            tmp_dist = distance_between(rep_a[2], rep_b[2])

            if tmp_dist > furthest_dist:
                furthest_dist = tmp_dist
                furthest = (i, j)

    return furthest

def add_centroid(data, centroids):
    """Returns the representative to use as the new centroid.

    Parameters:
    - Data: the list of voting data
    - Centroids: the indeces of the representatives currently 
                 being used as centroids.

    Returns: The index of the new representative to turn into a centroid.
    """
    furthest = 0
    furthest_dist = 0

    for i, rep in enumerate(data):
        tmp_dist = 0
        
        if i in centroids:
            continue

        for cent_rep in centroids:
            tmp_dist += distance_between(rep[2], data[cent_rep][2]) 

        if tmp_dist > furthest_dist:
            furthest = i
            furthest_dist = tmp_dist

    return furthest


def get_num_val(vote):
    """Converts a single vote to a numerical value.

    Parameters:
    - vote (char): The vote to convert.

    Returns: 1 for +, 0 for . and -1 for -.
    """
    if type(vote) == int or type(vote) == float:
        return vote
    if vote == '+':
        return 1
    if vote == '.':
        return 0
    if vote == '-':
        return -1

    print(vote)
    print("ERROR! invalid vote value")
    sys.exit()

def find_closest_centroid(centroids, vote_history):
    """Finds the closest centroid to the rep.

    Parameters:
    - centroids: a list of centroid vectors
    - vote_history: the voting history of the rep

    Returns: the index of the closest centroid
    """
    closest_dist = 9999999999999
    closest_centroid = 0

    for i, centroid in enumerate(centroids):
        dist = distance_between(centroid, vote_history)
        if dist < closest_dist:
            closest_dist = dist
            closest_centroid = i

    return closest_centroid

def add_vectors(vector1, vector2):
    """Adds and returns the two vectors in feature space.

    Parameters: The two vectors to add
   
    Returns: The sum of the two vectors
    """

    vec_sum = []

    for i,_ in enumerate(vector1):
        #TODO this is inefficient
        tmp_sum = get_num_val(vector1[i]) + get_num_val(vector2[i])
        vec_sum.append(tmp_sum)

    return vec_sum[:]

def calc_center(data, group):
    """Calculates the center of a group of representatives

    Parameters:
    - data: The raw voting data
    - group: A list of indexes of reps in the group

    Returns: The center position of the group
    """
    position_sum = [0 for x in data[0][2]]

    for rep in group:
        position_sum  = add_vectors(position_sum, data[rep][2])

    return [x/len(group) for x in position_sum]

def kmeans_cycle(data, centroids):
    """Performs cycles of kmeans until the system stabilizes.

    Parameters:
    - Data: the voting data
    - Centroids: The list of centroids in vector form

    Returns: The groups of reps about each centroid
    """

    # make a list of groups
    groups = [[] for x in centroids]

    # Assign reps to each group
    for i, rep in enumerate(data):
        closest_centroid = find_closest_centroid(centroids, rep[2])
        groups[closest_centroid].append(i)

    # Move the centroids
    new_centroids = [calc_center(data, group) for group in groups]

    # repeat if the centroids are different
    if not are_equal(centroids, new_centroids):
        return kmeans_cycle(data, new_centroids)
    else:
        return groups


def print_group_stats(data, groups):

    for i,group in enumerate(groups):
        size = len(group)
        percent_d = len([1 for x in group if data[x][1] == 'D'])/size
        percent_d*=100
        percent_r = 100-percent_d

        print(f"\tGroup {i+1}: size {size} ({percent_d:.3f}% D, {percent_r:.3f}% R)")


def are_equal(vector1, vector2):
    """Returns True if the vectors are equal, false otherwise"""
    for i,_ in enumerate(vector1):
        if vector1[i] != vector2[i]:
            return False

    return True


if __name__ == '__main__':

    # Check that the file exists
    if not os.path.exists(sys.argv[1]):
        print("kmeans: This file does not exist")
        sys.exit()

    if int(sys.argv[2]) < 2:
        print("kmeans: Must have use two or more centroids")
        sys.exit()

    # Read the data set line by line
    file = sys.argv[1]
    data = []
    with open(file) as f:
        data = f.readlines()
    f.close()

    # Split each line by white space
    data = [line.split() for line in data]

    # Choose initial centroids
    rep_a, rep_b = get_furthest_apart(data)

    # list of initial centroids
    init_centroids = [rep_a, rep_b]
    while len(init_centroids) < int(sys.argv[2]):
        init_centroids.append(add_centroid(data, init_centroids))

    # print out the initial centroids
    print("Initial centroids based on: ", end="")
    for rep in init_centroids:
        print(data[rep][0] + " ", end="")
    print("")

    # Turn the intitial centroids into a list of vectors
    centroids = [convert_to_nums(data[x][2]) for x in init_centroids]
    groups = kmeans_cycle(data, centroids)
    print_group_stats(data, groups)
