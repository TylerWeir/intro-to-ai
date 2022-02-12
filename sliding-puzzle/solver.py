"""solver.py

Uses the A* search algorithm to solve a given m x n sliding puzzle.

Written by Tyler Weir
02/09/2022"""

import heapq

class State:
    """Represents a state of a sliding puzzle."""

    def __init__(self, width, height, layout):
        self.width = width
        self.height = height

        # Flatten the layout if it came as a 2D array
        if type(layout[0]) == type([]):
            # Functional way to flatten a 2D list courtesy of
            # geeksforgeeks.com.
            self.state = [j for sub in layout for j in sub]
        else:
            self.state = layout


    def is_solveable(self):
        """Returns True if the puzzle is possible to solve.
        False otherwise."""

        # Remove the zero
        scratch = [x for x in self.state if x != 0]

        # Count the total inversions.
        inversions = 0
        for i in range(len(scratch)):
            for j in range(len(scratch)-i):
                if scratch[i+j] < scratch[i]:
                    inversions += 1

        # Solveable if the width is odd and the inversions is even.
        if self.width % 2 != 0 and inversions % 2 == 0:
            return True

        # The number of rows from the bottom 0 is at.
        zero_row = self.height - 1 - self.state.index(0) // self.width

        # Solveable if the sum of zero_row and inversions is even
        return (zero_row + inversions) % 2 == 0

    def get_moves(self):
        """Returns a tuple of the states representing the states
        possible after making a single legal move."""
        #TODO: Fill in
    
    def __str__(self):
        return str(self.state)

    def __eq__(self, other):
        """Overrides the built-in equals method."""
        same_inst = isinstance(other, State)
        if not same_inst:
            return False

        same_width = self.width == other.width
        same_height = self.height == other.height
        same_state = tuple(self.state) == tuple(other.state)
        return same_width and same_height and same_state

    def __hash__(self):
        """Overrides the built-in hash method."""
        tmp = (* self.state, self.width, self.height)
        return hash(tmp)

class Priority_Queue:
    """A min priority queue."""
    def __init__(self):
        self.entries = []

    def is_empty(self):
        """Returns True if the queue is empty."""
        return len(self.entries) == 0

    def insert(self, entry):
        """Add and element to the pq."""
        self.entries.append(entry)
        heapq.heapify(self.entries)

    def pop(self):
        """Pops the element with the highest priority."""
        item = self.entries.pop(0)
        heapq.heapify(self.entries)
        return item[1]

    def contains(self, item):
        """Returns true if the item is contained in the set"""
        for entry in self.entries:
            if entry[1] == item:
                return True

        return False

    def __str__(self):
        return str(self.entries)



def __calc_h(state:State):
    """Returns the admissible heuristic of the state.

    Heuristic is calculated by relaxing the constraints of the
    puzzle and allowing tiles to move through one another.
    For each tile, the number of rows and columns it is away
    from its goal position is summed.

    Note this heuristic works for small puzzles. It is not
    accurate enough for large puzzles.
    """

    sum_moves = 0

    for i in range(1, state.width * state.height):

        index = state.state.index(i)

        # TODO: verify that these are right
        # First find number of vertical moves
        row = index // state.width
        goal_row = i // state.width

        # Then find number of horizontal moves
        column = index // state.height
        goal_column = i // state.height

        sum_moves += abs(goal_row - row)
        sum_moves += abs(goal_column - column)

    return sum_moves


def solve(puzzle):
    """Solve a sliding puzzle.
    Keyword arguments:
    puzzle -- A 2D row-major list representing a puzzle.

    Solves the puzzle using the a-star search algorithm and the
    heuristic.

    defined in the state class.
    """
    print(puzzle)
    start_state = State(len(puzzle[0]), len(puzzle), puzzle)

    if not start_state.is_solveable():
        return None
    return __solve(start_state)

def __solve(start_state):
    """Private solver using A* to search through the states.

    Keyword arguments:
    start_state -- A SOLVABLE State"""

    # Setup
    open_list = Priority_Queue()
    open_list.insert((1, start_state))
    came_from = {}

    g_score = {}
    g_score.update({start_state:0})

    f_score = {}
    f_score.update({start_state: __calc_h(start_state)})

    while not open_list.is_empty():
        current = open_list.pop()

        # Check if current is the goal
        if __calc_h(current) == 0:
            return __make_path(came_from, current)

        for child in current.get_moves():
            # Child is one move away
            child_g_score = g_score.get(current)+1
            if g_score.get(child) == None or child_g_score < g_score.get(child):
                # This is the best path
                came_from.update({child, current})
                g_score.update({child: child_g_score})
                f_score.update({child: child_g_score + __calc_h(child)})

                if not open_list.contains(child):
                    open_list.add((f_score.get(child), child))
    return None

def __make_path(came_from, goal):
    """Returns the moves taken to get from the starting state to the 
    goal state."""
    return ['U', 'D']

if __name__ == "__main__":
    #myState = State(4, 4, [6, 5, 2, 0, 3, 7, 11, 4, 9, 1, 10, 8, 15, 14, 13, 12])
    myState = State(4, 3, [1, 2, 3, 4, 0, 5, 6, 7, 9, 10, 11, 8])
    print(f'The solution: {myState.is_solveable()}')

    """
    myqueue = Priority_Queue()
    myqueue.insert((3, 'apple'))
    myqueue.insert((2, 'pear'))
    myqueue.insert((5, 'orange'))
    myqueue.insert((11, 'red'))

    print("list is " + str(myqueue))

    print("popped: "+ str(myqueue.pop()))
    myqueue.insert((1, "YAYA"))
    print("list is " + str(myqueue)) """
