"""solver.py

Uses the A* search algorithm to solve a given m x n sliding puzzle.

Written by Tyler Weir
02/09/2022"""

from goto import goto, label

class State:
    """Represents a state of a sliding puzzle."""

    def __init__(self, width, height, layout):
        self.f_score = -1
        self.h_score = self.calc_heuristic()
        self.g_score = -1
        self.width = width
        self.height = height

        # Flatten the layout if it came as a 2D array
        if type(layout[0]) == type([]):
            # Functional way to flatten a 2D list courtesy of
            # geeksforgeeks.com.
            self.state = [j for sub in layout for j in sub]
        else:
            self.state = layout

    def calc_heuristic(self):
        """Returns the admissible heuristic of the state.

        Heuristic is calculated by relaxing the constraints of the
        puzzle and allowing tiles to move through one another.
        For each tile, the number of rows and columns it is away
        from its goal position is summed.

        Note this heuristic works for small puzzles. It is not
        accurate enough for large puzzles.
        """

        sum_moves = 0

        for i in range(1, self.width * self.height):

            index = self.state.index(i)

            # TODO: verify that these are right
            # First find number of vertical moves
            row = index // self.width
            goal_row = i // self.width

            # Then find number of horizontal moves
            column = index // self.height
            goal_column = i // self.height

            sum_moves += abs(goal_row - row)
            sum_moves += abs(goal_column - column)

        return sum_moves

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

def solve(puzzle):
    """Solve a sliding puzzle.
    Keyword arguments:
    puzzle -- A 2D row-major list representing a puzzle.

    Solves the puzzle using the a-star search algorithm and the
    heuristic.

    defined in the state class.
    """
    state = State(len(puzzle), len(puzzle[0]), puzzle)

    if not state.is_solveable():
        return None
    else:
       return ['U', 'D', 'U', 'D', 'U', 'D'] 

def __solve(start_state):
    """Private solver using A* to search through the states.

    Keyword arguments:
    start_state -- A SOLVABLE State"""

    #TODO: Implement A*
    open_list = {}
    closed_list = {}

    # Put the start node on the open list with f = h
    open_list.update({start_state:None})
    start_state.f_score = start_state.h_score

    while len(open_list.keys()) > 0:
        #TODO: When to update h and g and f?
        current = __pop_smallest(open_list)
        if current[0].h_score == 0:
            # We found the solution
            break

        # Generate the sucessor states
        moves = current[0].get_moves()

        for move in moves:
            #set current move cost
            current_cost = 3#TODO

            # Is the move on the open list?
            if open_list.get(move) != None:
                if move.g_score <= current_cost:
                    goto .add_to_closed_list
            # Is the move on the closed list?
            elif close_list.get(move) != None:
                if move.g_score <= current_cost:
                    goto .add_to_closed_list
                else:
                    # move the move from the closed list to
                    # the open list
                    closed_list.pop(move)
                    open_list.update({move:current[0]})

            else:
                # Add move to the open list
                open_list.update({move: current[0]})
                # TODO set move.h_score
            move.g_score = current_cost

        label .add_to_closed_list
        closed_list.update(current[0], current[1])
   # should be done
   # TODO return the closed list?

def __pop_smallest(dictionary):
    """Returns the state with the smallest f_score as a tuple
    with the parent state."""
    items = dictionary.items()
    lowest_state = items[0]

    # Iterate through all the states in the list
    for (state, parent) in items:
        if state.f_score < lowest_state.f_score:

            lowest_state = (state, parent)

    # Return the lowest.
    dictionary.pop(lowest_state[0])
    return lowest_state


if __name__ == "__main__":
    #myState = State(4, 4, [6, 5, 2, 0, 3, 7, 11, 4, 9, 1, 10, 8, 15, 14, 13, 12])
    myState = State(4, 3, [1, 2, 3, 4, 0, 5, 6, 7, 9, 10, 11, 8])
    print(f'The solution: {myState.solve()}')
