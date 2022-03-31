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

class Node:
        """Represents a Node in the decision tree. Terminal Nodes
        have no children and thus must contain a label."""

        def __init__(self, vote, value, parent=None):
            """Creates a new node."""
            self.parent = parent
            self.children = [] 
            self.vote = vote
            self.value = value

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

        def __str__(self):
            return f"(NODE: {self.vote} {self.value})"


class DecisionTree:
    """A Decision Tree classifier."""

    def __init__(self):
        """Default constructor for making a new decision tree."""
        self.root = Node("Root", "ISSUE")

    def build_tree_from_data(self, data, node):
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

            if gain > greatest_gain:
                greatest_gain = gain
                best_feature = i

        node.value = f"issue {best_feature}"
        
        if greatest_gain == 0:
            #TODO
            node.value = 'R'
            return
        
        # Checks:

        # IF both Democrat and Republican with same voting record
        # create a leaf that classifies them as the majority

        # If a branch has zero reps classify based on the majority at its parent
        # keep going up the tree until you have a majority
        
        # Third, Choose the single best feature and divide the data
        # set into two or more discrete groups.

        yes_votes = [(n, l, v) for (n, l, v) in data if v[best_feature] == '+']
        no_votes = [(n, l, v) for (n, l, v) in data if v[best_feature] == '-']
        ab_votes = [(n, l, v) for (n, l, v) in data if v[best_feature] == '.']

        # Fourth, if a subgroup is not uniformly labeled, recurse
        """
        if len(yes_votes) != 0:
            yes_node = Node('+', "Issue", parent=node)
            node.add_child(yes_node)
                
            if not self.__is_uniform(yes_votes):
                self.build_tree_from_data(yes_votes, yes_node)            

        if len(no_votes) != 0:
            no_node = Node('-', "Issue", parent=node)
            node.add_child(no_node)

            if not self.__is_uniform(no_votes):
                self.build_tree_from_data(no_votes, no_node)            
        
        if len(ab_votes) != 0:
            ab_node = Node('.', "Issue", parent=node)
            node.add_child(ab_node)

            if not self.__is_uniform(ab_votes):
                self.build_tree_from_data(ab_votes, ab_node)            
        """
        if len(yes_votes) != 0:
            if self.__is_uniform(yes_votes):
                # make a leaf node
                yes_node = Node('+', 'R', parent=node)
                node.add_child(yes_node)

            else:
                yes_node = Node('+', 'Some Issue', parent=node)
                node.add_child(yes_node)

                # recurse on the yes voters
                self.build_tree_from_data(yes_votes, yes_node)
        else:
            # There are zero reps
            yes_node = Node('+', 'R', parent=node)
            node.add_child(yes_node)

        if len(no_votes) != 0:
            if self.__is_uniform(no_votes):
                # make a leaf node
                no_node = Node('-', 'R', parent=node)
                node.add_child(no_node)

            else: 
                # recurse on the no voters
                no_node = Node('-', 'Some Issue', parent=node)
                node.add_child(no_node)

                self.build_tree_from_data(no_votes, no_node)
        else:
            # There are zero reps
            # TODO:
            # currently just assigning 'R'
            no_node = Node('-', 'R', parent=node)
            node.add_child(no_node)
            

        if len(ab_votes) != 0:
            if self.__is_uniform(ab_votes):
                # make a leaf node
                ab_node = Node('.', 'R', parent=node)
                node.add_child(ab_node)
            else:
                ab_node = Node('.', 'Some Issue', parent=node)
                node.add_child(ab_node)

                # recurse on the ab voters
                self.build_tree_from_data(ab_votes, ab_node)
        else: 
            # There are zero reps
            ab_node = Node('.', 'R', node)
            node.add_child(ab_node)

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

        if len(data) == 0:
            return True

        test = data[0][1]

        for item in data:
            if item[1] != test:
                return False
        
        return True

    def __persistant_get_majority(self, node):
        pass

    def __get_majority(self, data):
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

        
    def print_tree(self, root, depth):
        """Prints a text representation of the decision tree."""

        if depth == 0:
            print(f'{root.value}:')

            # Recurse on children
            for child in root.children:
                self.print_tree(child, depth + 1)

        else:
            # Print the node
            spaces = ""
            for i in range(depth):
                spaces += "  "

            print(spaces + root.vote + " " + root.value)

            # Recurse on children
            for child in root.children:
                self.print_tree(child, depth + 1)


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
    
    my_tree.build_tree_from_data(data, my_tree.root)

    """
    print(len(my_tree.root.children))
    
    print("ROOT Children: ")
    for child in my_tree.root.children:
        print(child)

    """
    
    # Display the tree
    my_tree.print_tree(my_tree.root, 0)


""" if len(yes_votes) != 0:
            if self.__is_uniform(yes_votes):
                # make a leaf node
                yes_node = self.Node('+', 'R', parent=node)
                node.children.append(yes_node)

            else:
                yes_node = self.Node('+', 'Some Issue', parent=node)
                node.children.append(yes_node)

                # recurse on the yes voters
                self.build_tree_from_data(yes_votes, yes_node)
        else:
            # There are zero reps
            yes_node = self.Node('+', 'R', parent=node)
            node.children.append(yes_node)

        if len(no_votes) != 0:
            if self.__is_uniform(no_votes):
                # make a leaf node
                no_node = self.Node('-', 'R', parent=node)
                node.children.append(no_node)

            else: 
                # recurse on the no voters
                no_node = self.Node('-', 'Some Issue', parent=node)
                node.children.append(no_node)

                self.build_tree_from_data(no_votes, no_node)
        else:
            # There are zero reps
            # TODO:
            # currently just assigning 'R'
            no_node = self.Node('-', 'R', parent=node)
            node.children.append(no_node)
            

        if len(ab_votes) != 0:
            if self.__is_uniform(ab_votes):
                # make a leaf node
                ab_node = self.Node('.', 'R', parent=node)
                node.children.append(ab_node)
            else:
                ab_node = self.Node('.', 'Some Issue', parent=node)
                node.children.append(ab_node)

                # recurse on the ab voters
                self.build_tree_from_data(ab_votes, ab_node)
        else: 
            # There are zero reps
            ab_node = self.Node('.', 'R', node)
            node.children.append(ab_node)

        print(f"{node} has {len(node.children)} children")"""