# solver.py
#
# Uses the A* search algorithm to solve a given m x n sliding puzzle.  
#
# Written by Tyler Weir
# 02/07/2022

class State:

    def __init__(self, m, n, starting_state):
        self.m = m
        self.n = n
        if len(starting_state) > 0:
            # Functional way to flatten a 2D list courtesy of geeksforgeeks.com. 
            self.state = [j for sub in ini_list for j in sub]  
        else:
            self.state = starting_state

    def print(self):
        print(str(self.tester))

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
        return isinstance(other, Person) and self.m == other.m and self.n == other.n and tuple(self.state) == tuple(other.state)

    def __hash__(self):
        # TODO: This works for now...
        return hash(tuple(self.state))
        


def solve(puzzle):
    """Solve a sliding puzzle.

    Keyword arguments:
    puzzle -- A 2D row-major list representing a puzzle. 
    """
    state = State(len(puzzle), len(puzzle[0]), puzzle)

if __name__ == "__main__":
    myState = State(4, 4, [6, 5, 2, 0, 3, 7, 11, 4, 9, 1, 10, 8, 15, 14, 13, 12])
    print(f'The puzzle is solveable: {myState.isSolveable()}')
