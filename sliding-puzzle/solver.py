"""solver.py

Uses the A* search algorithm to solve a given m x n sliding puzzle.

Written by Tyler Weir
02/12/2022"""

import heapq

class State:
    """Represents a state of a sliding puzzle."""

    def __init__(self, width, height, layout):
        """Creates a new State.
        width: The integer width of the puzzle.
        height: The integer height of the puzzle.
        layout: Either a 1D or 2D list describing the layout of the
                puzzle."""
        self.width = width
        self.height = height

        self.g = -1
        self.h = -1
        self.f = -1

        self.parent = None

        # Flatten the layout if it came as a 2D array
        if type(layout[0]) == type([]):
            # Functional way to flatten a 2D list courtesy of
            # geeksforgeeks.com.
            self.state = [j for sub in layout for j in sub]
        else:
            self.state = layout

    def is_solveable(self):
        """Returns True if it is possible to solve a puzzle in this
        state, False otherwise."""

        # Remove the zero
        scratch = [x for x in self.state if x != 0]

        # Count the total inversions.
        inversions = 0
        for i, _ in enumerate(scratch):
            for j in range(len(scratch) - i):
                if scratch[i+j] < scratch[i]:
                    inversions += 1

        # Solveable if the width is odd and the inversions is even.
        if self.width % 2 != 0 and inversions % 2 == 0:
            return True

        # The number of rows from the bottom 0 is at.
        zero_row = self.height - 1 - self.state.index(0) // self.width

        # Solveable if the sum of zero_row and inversions is even
        return (zero_row + inversions) % 2 == 0

    def __zero_row(self):
        """Returns the row that the zero is in."""
        index = self.state.index(0)
        return index//self.width

    def __zero_column(self):
        """Returns the columnt that the zero is in."""
        index = self.state.index(0)
        return index%self.width

    def get_moves(self):
        """Returns a tuple of the states representing the states
        possible after making a single legal move."""
        zero_row = self.__zero_row()
        zero_col = self.__zero_column()

        is_top = zero_row == 0
        is_bottom = zero_row == self.height-1
        is_left = zero_col == 0
        is_right = zero_col == self.width-1

        moves = []

        # Make all avaliable moves
        if not is_top:
            moves.append(self.__move_down())

        if not is_bottom:
            moves.append(self.__move_up())

        if not is_right:
            moves.append(self.__move_left())

        if not is_left:
            moves.append(self.__move_right())

        return moves

    def __move_up(self):
        """Returns the new state after moving up."""
        child = State(self.width, self.height, self.state[:])
        zero_index = child.state.index(0)
        child.swap(zero_index,zero_index+child.width)
        return child

    def __move_down(self):
        """Returns the new state after moving down."""
        child = State(self.width, self.height, self.state[:])

        zero_index = child.state.index(0)
        child.swap(zero_index,zero_index-child.width)
        return child

    def __move_left(self):
        """Returns the new state after moving left."""
        child = State(self.width, self.height, self.state[:])
        zero_index = child.state.index(0)
        child.swap(zero_index,zero_index+1)
        return child

    def __move_right(self):
        """Returns the new state after moving right."""
        child = State(self.width, self.height, self.state[:])
        zero_index = child.state.index(0)
        child.swap(zero_index,zero_index-1)
        return child

    def swap(self, index_one, index_two):
        """Swaps elements in the internal state list."""
        tmp = self.state[index_one]
        self.state[index_one] = self.state[index_two]
        self.state[index_two] = tmp


    def __str__(self):
        """Overrides the built-in string method."""
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
        #TODO: Can I make this faster? Maybe just store state as 
        # a tuple already?
        tmp = (* self.state, self.width, self.height)
        return hash(tmp)

class PriorityQueue:
    """A minimum priority queue to act as the open list in A*. It is
    dictionary backed for lighing fast methods."""

    def __init__(self):
        """Creates a new empty PriorityQueue."""
        self.priorities = {}
        self.size = 0
        self.entries = []
        self.entry_num = 0

    def is_empty(self):
        """Returns True if the queue is empty, False otherwise.
        Time Complexity: O(1)"""
        return self.size == 0

    def get_size(self):
        """Returns the size of the priority queue.
        Time Complexity: O(1)"""
        return self.size

    def insert(self, priority, state):
        """Add and element to the priority queue.
        Time Complexity: O(n)"""

        # Update counters
        self.entry_num += 1
        self.size += 1

        # Update list and dictionary
        self.priorities.update({state:priority})
        self.entries.append((priority, self.entry_num, state))
        heapq.heapify(self.entries)

    def pop(self):
        """Pops the element with the least f score.
        Time Complexity: O(n)"""

        # Clear from list and dictionary.
        self.priorities.pop(self.entries[0][2])
        item = self.entries.pop(0)
        self.size -= 1

        # Make the heap again.
        heapq.heapify(self.entries)
        return item[2]

    def contains(self, item):
        """Returns True if the item is in the queue, False otherwise.
        Time Complexity: O(1)"""
        return item in self.priorities

    def get_priority(self, state:State):
        """Get the priority of a state in the queue. Returns -1 if
        the item could not be found.
        Time Complexity: O(1)"""
        return self.priorities.get(state, -1)

    def remove(self, state:State):
        """Remove a state from the queue.
        Time Complexitu: O(n)"""
        for i, entry in enumerate(self.entries):
            if entry[2] == state:

                self.size -= 1
                self.priorities.pop(state)
                self.entries.pop(i)
                break

        heapq.heapify(self.entries)

    def __str__(self):
        """Returns a string representation of the queue."""
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

        # First find number of vertical moves
        row = index // state.width
        goal_row = (i-1) // state.width

        # Then find number of horizontal moves
        column = index % state.width
        goal_column = (i-1) % state.width

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
    start_state = State(len(puzzle[0]), len(puzzle), puzzle)

    if not start_state.is_solveable():
        return None
    return __solve(start_state)

def __solve(start_state):
    """Private solver using A* to search through the states."""

    # Setup lists
    open_list = PriorityQueue()
    closed_list = {}

    # Add starting node to the open list.
    open_list.insert(1, start_state)

    # Set the scores for the first state
    start_state.g = 0
    start_state.h = __calc_h(start_state)
    start_state.f = start_state.h

    start_state.parent = None

    while not open_list.is_empty():

        # Get the state with the lowest f score from the open list.
        current = open_list.pop()

        # Is this state the goal?
        if current.h == 0:
            closed_list.update({current: current.parent})
            return __make_path(closed_list, current)

        # Explore the child states of the current state.
        # Add them to the open list if they aren't already
        # there or if they score better.
        for child in current.get_moves():

            # calculate the child's scores
            child.g = current.g + 1
            child.h = __calc_h(child)
            child.f = child.g + child.h

            child.parent = current

            # Is the child on the closed list?
            if child in closed_list:
                continue

            # Is the child on the open list?
            priority = open_list.get_priority(child)
            if priority != -1:
                # Is the priority child better?
                if priority <= child.f:
                    continue
                # This child is better
                open_list.remove(child)
                open_list.insert(child.f, child)
            else:
                # The child is not on the open list
                open_list.insert(child.f, child)

        # Add the explored node to the closed list
        closed_list.update({current: current.parent})

    print("ERROR: ran the open list dry.")


def __decode_move(before:State, after:State):
    """Returns the move that was made to go from the before state
    to the after state."""
    before_zero = before.state.index(0)
    after_zero = after.state.index(0)

    diff = after_zero - before_zero

    if diff == before.width:
        return 'D'
    if diff == -before.width:
        return 'U'
    if diff == 1:
        return 'R'
    if diff == -1:
        return 'L'

    print("ERROR making path")
    return None


def __make_path(came_from, goal):
    """Returns the moves taken to get from the starting state to the
    goal state."""

    moves = []
    current = goal
    parent = came_from.get(current)

    # Keeping interating through parents until we reach the
    # starting state.
    while parent:
        moves.insert(0, __decode_move(current, parent))
        current = came_from.get(current)
        parent = came_from.get(current)

    return moves
