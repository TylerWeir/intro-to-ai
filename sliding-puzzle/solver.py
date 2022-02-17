"""solverpy

Uses the A* search algorithm to solve a given m x n sliding puzzle.

Written by Tyler Weir
02/12/2022"""

import sys
import heapq
import threading
from queue import PriorityQueue

# Global vars for the thread to accesss
mutex = threading.Lock()
claimed = {}
closed_list_1 = {}
closed_list_2 = {}
finished = False
connection = None

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
            self.state = layout[:]

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

def __calc_h(state_1:State, state_2:State):
    """Returns the admissible heuristic of the state.

    Heuristic is calculated by relaxing the constraints of the
    puzzle and allowing tiles to move through one another.
    For each tile, the number of rows and columns it is away
    from its goal position is summed.

    Note this heuristic works for small puzzles. It is not
    accurate enough for large puzzles.
    """

    sum_moves = 0
    
    width = state_1.width
    height = state_1.height

    for i in range(1, width * height):
        
        index_1 = state_1.state.index(i)
        index_2 = state_2.state.index(i)

        # First find number of vertical moves
        row_1 = index_1 // width
        row_2 = index_2 // width

        # Then find number of horizontal moves
        column_1 = index_1 % width
        column_2 = index_2 % width

        sum_moves += abs(row_1 - row_2)
        sum_moves += abs(column_1 - column_2)

    return sum_moves

def solve(puzzle):
    """Solve a sliding puzzle.
    Keyword arguments:
    puzzle -- A 2D row-major list representing a puzzle.

    Solves the puzzle using the a-star search algorithm and the
    heuristic.

    defined in the state class.
    """
    width = len(puzzle[0])
    height = len(puzzle)

    solved = list(range(1, width*height))
    solved.append(0)
    
    # p1 is working from start to solved
    p1_start_state = State(width, height, puzzle)
    p1_goal_state = State(width, height, solved)

    # p2 is working from solved to start
    p2_start_state = State(width, height, solved)
    p2_goal_state = State(width, height, puzzle)

    p1 = threading.Thread(target=p1_solve, args=(p1_start_state, p1_goal_state))
    p2 = threading.Thread(target=p2_solve, args=(p2_start_state, p2_goal_state))

    # Start the threads
    p1.start()
    p2.start()
    print("Started threads")

    # Wait for the threads to come back
    p1.join()
    p2.join()
    print("Threads returned")

    global closed_list_1
    global closed_list_2
    global connection

    print(f"connection is in cl1 {connection in closed_list_1}")
    print(f"connection is in cl2 {connection in closed_list_2}")

    return __make_path(p1_start_state, p1_goal_state)

def is_claimed_by(state, thread_num):
    """Checks if a thread has put a state on it's closed_list."""
    global mutex
    global claimed

    mutex.acquire()

    if claimed.get(state,-1) == thread_num:
        mutex.release()
        return True

    mutex.release()
    return False

def add_if_missing(state, thread_num, opposite_thread):
    """Adds a state to the public closed_list record. Returns True
    if successfully added, False otherwise.

    thread_num - The identity of the thread we are looking out for."""
    global mutex
    global claimed

    mutex.acquire()

    if claimed.get(state, -1) == opposite_thread:
        mutex.release()
        return False

    claimed.update({state : thread_num})
    mutex.release()
    return True

def p1_solve(start_state, goal_state):
    """The solver to be used by p1."""

    # Access the global vars
    global finished
    global discovered
    global connection
    global closed_list_1

    # Setup the lists
    open_list = PriorityQueue()
    entrynum = 1
    closed_list_1 = {}
    f = {}

    # Get the scores for the start state
    start_state.g = 0
    start_state.h = __calc_h(start_state, goal_state)
    start_state.f = start_state.h

    start_state.parent = None

    # Add start state to the openlist.
    open_list.put((start_state.f, entrynum, start_state))
    entrynum += 1

    # Enter the search
    while not finished:
        
        # Get the next best looking state
        current = open_list.get()[2]

        # Check if this state is on the closed list
        if current in closed_list_1:
            continue

        # Check if this state has been added to the other
        # thread's closed list
        if is_claimed_by(current, 2):
            # This is our connection!
            closed_list_1.update({current: current.parent})
            finished = True
            print(f"connection is {current}")
            if connection == None:
                connection = current
            return

        # Explore the child states of the current. Add them to the
        # open_list if they are new or good, otherwise skip them.
        for child in current.get_moves():

            # Make sure we need this one
            if child in closed_list_1:
                continue

            # calculate the child's scores
            child.g = current.g + 1
            child.h = __calc_h(child, goal_state)
            child.f = child.g + child.h
            child.parent = current

            # have we seen better?
            best_score = f.get(child, sys.maxsize)
            if best_score <= child.f:
                continue

            #Otherwise this is the best
            f.update({child: child.f})
            open_list.put((child.f, entrynum, child))
            entrynum += 1

        # Add the explored node to the public closed list
        if not add_if_missing(current, 1, 2):
            # Adding failed!
            # That means this is a connection.
            finished = True
            closed_list_1.update({current: current.parent})
            print(f"connection is {current}")
            if connection == None:
                connection=current
            return

        # Otherwise continue as usual
        f.update({current: current.f})
        closed_list_1.update({current: current.parent})

def p2_solve(start_state, goal_state):
    """The solver to be used by p2."""

    # Access the global vars
    global finished
    global discovered
    global connection
    global closed_list_2

    # Setup the lists
    open_list = PriorityQueue()
    entrynum = 0
    closed_list_2 = {}
    f = {}

    # Get the scores for the start state
    start_state.g = 0
    start_state.h = __calc_h(start_state, goal_state)
    start_state.f = start_state.h

    start_state.parent = None

    # Add start state to the openlist.
    open_list.put((start_state.f, entrynum, start_state))
    entrynum += 1

    # Enter the search
    while not finished:
        
        # Get the next best looking state
        current = open_list.get()[2]

        # Check if this state is on the closed list
        if current in closed_list_2:
            continue

        # Check if this state has been added to the other
        # thread's closed list
        if is_claimed_by(current, 1):
            # This is our connection!
            closed_list_2.update({current: current.parent})
            finished = True
            print(f"connection is {current}")
            if connection == None:
                connection = current
            return

        # Explore the child states of the current. Add them to the
        # open_list if they are new or good, otherwise skip them.
        for child in current.get_moves():

            # Make sure we need this one
            if child in closed_list_2:
                continue

            # calculate the child's scores
            child.g = current.g + 1
            child.h = __calc_h(child, goal_state)
            child.f = child.g + child.h
            child.parent = current

            # have we seen better?
            best_score = f.get(child, sys.maxsize)
            if best_score <= child.f:
                continue

            #Otherwise this is the best
            f.update({child: child.f})
            open_list.put((child.f, entrynum, child))
            entrynum+=1

        # Add the explored node to the public closed list
        if not add_if_missing(current, 2, 1):
            # Adding failed!
            # That means this is a connection.
            finished = True
            closed_list_2.update({current: current.parent})
            print(f"connection is {current}")
            if connection == None:
                connection = current
            return

        # Otherwise continue as usual
        f.update({current: current.f})
        closed_list_2.update({current: current.parent})

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


def __make_path(starte_state, goal_state):
    """Returns the moves taken to get from the starting state to the
    goal state."""

    global connection
    global closed_list_1
    global closed_list_2

    moves = []
    current = connection

    parent = closed_list_1.get(current)

    # Keeping interating through parents until we reach the
    # starting state.
    while parent:
        moves.insert(0, __decode_move(current, parent))
        current = closed_list_1.get(current)
        parent = closed_list_1.get(current)

    current = connection
    parent = closed_list_2.get(current)

    # Keeping interating through parents until we reach the
    # starting state.
    while parent:
        moves.append(__decode_move(parent, current))
        current = closed_list_2.get(current)
        parent = closed_list_2.get(current)

    return moves
