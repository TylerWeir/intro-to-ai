# solver.py
#
# Uses the A* search algorithm to solve a given m x n sliding puzzle.  
#
# Written by Tyler Weir
# 02/07/2022

class State:

    def __init__(self, m, n, layout):

        self.h = self.calc_heuristic()
        self.g   #TODO: IS this needed:
        self.m = m
        self.n = n

        # Flatten the layout if it came as a 2D array 
        if type(layout[0]) == type([]) :
            # Functional way to flatten a 2D list courtesy of geeksforgeeks.com. 
            self.state = [j for sub in layout for j in sub]  
        else:
            self.state = layout 

    def calc_heuristic(self):
        """Returns the admissible heuristic of the state.
        
        Heuristic is calculated by relaxing the constraints of the puzzle 
        and allowing tiles to move through one another.  For each 
        tile, the number of rows and columns it is away from its goal position
        is summed. 
        
        Note this heuristic works for small puzzles. It is not accurate enough
        for large puzzles.
        """

        sum_moves = 0

        for i in range(1, self.m * self.n):

            index = self.state.index(i)
            
            # TODO: verify that these are right
            # First find number of vertical moves
            row = index // self.m
            goalRow = i // self.m
            
            # Then find number of horizontal moves
            column = index // self.n 
            goalColumn = i // self.n 

            sum_moves += abs(goalRow - row) + abs(goalColumn - column) 

        return sum_moves

    def isSolveable(self):
        """Returns True if the puzzle is possible to solve. False otherwise."""
        
        # Remove the zero
        scratch = [x for x in self.state if x != 0]

        # Count the total inversions.
        inversions = 0
        for i in range(len(scratch)):
            for j in range(len(scratch)-i): 
                if scratch[i+j] < scratch[i]: inversions += 1 
        
        # Solveable if the width is odd and the inversions is even.
        if self.m % 2 != 0 and inversions % 2 == 0: return True

        # The number of rows from the bottom 0 is at. 
        zero_row = self.n - 1 - self.state.index(0) // self.m

        # Solveable if the sum of zero_row and inversions is even
        if (zero_row + inversions) % 2 == 0: return True
        else: return False

    def __eq__(self, other):
        return isinstance(other, State) and self.m == other.m and self.n == other.n and tuple(self.state) == tuple(other.state)

    def __hash__(self):
        # TODO: This works for now...
        return hash(tuple(self.state))
        
def solve(puzzle):
    """Solve a sliding puzzle.

    Keyword arguments:
    puzzle -- A 2D row-major list representing a puzzle. 

    Solves the puzzle using the a-star search algorithm and the heuristic 
    defined in the state class.
    """
    state = State(len(puzzle), len(puzzle[0]), puzzle)

if __name__ == "__main__":
    myState = State(4, 4, [6, 5, 2, 0, 3, 7, 11, 4, 9, 1, 10, 8, 15, 14, 13, 12])
    print(f'The puzzle is solveable: {myState.isSolveable()}')
