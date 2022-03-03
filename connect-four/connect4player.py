"""
This Connect Four player using the minimax algorithm optimized with
alpha-beta pruning to dominate all human adversaries, including its
creator.
"""
__author__ = "Tyler Weir"
__license__ = "MIT"
__date__ = "March 3, 2022"

import time

class ComputerPlayer:
    """
    This class represents a computer player in the connect4
    game. It uses the minimax algorithm and is optimized with
    alpha-beta pruning.
    """

    def __init__(self, player_id, difficulty_level):
        """
        Constructor, takes a difficulty level (likely the # of plies
        to look ahead), and a player ID that's either 1 or 2 that
        tells the player what its number is.
        """
        self.player_id = player_id

        if self.player_id == 1:
            self.opponent_id = 2
        else:
            self.opponent_id = 1

        self.difficulty = difficulty_level

    def pick_move(self, rack):
        """
        Pick the move to make. It will be passed a rack with the
        current board layout, column-major. A 0 indicates no token is
        there, and 1 or 2 indicate discs from the two players. Column
        0 is on the left, and row 0 is on the bottom. It must return
        an int indicating in which column to drop a disc. The player
        current just pauses for half a second (for effect), and then
        chooses a random valid move.
        """
        time.sleep(0.5) # pause purely for effect

        # Use minimax to find the best move
        best_move = None
        best_move_score = -float('inf')
        for move in self.get_moves(rack, self.player_id):
            score = self.minimax(move, 5, False, -float('inf'), float('inf'))

            if score > best_move_score:
                best_move_score = score
                best_move = move

        if best_move == None:
            # This means I'm gonna lose
            # Just take the first move
            best_move = self.get_moves(rack, self.player_id)[0]

        # Decode the move from the board state
        for i, col in enumerate(rack):
            num_spaces = sum(x == 0 for x in col)
            num_space_move = sum(x == 0 for x in best_move[i])

            if num_spaces != num_space_move:
                #This is where the tile was played
                return i

        print("major error")

    def minimax(self, board_state, depth, max_player, alpha, beta):
        """Contains the minimax algorithm"""

        # Return score if this is the bottom.
        if depth == 0:
            return self.calc_heuristic(board_state)
        
        # Evaluate the max player
        if max_player:
            max_score = -float('inf')
            for move in self.get_moves(board_state, self.player_id):
                score = self.minimax(move, depth-1, False, alpha, beta)
                max_score = max(score, max_score)
                alpha = max(alpha, max_score)
                if beta <= alpha:
                    break
            return max_score
        
        # Evaluate the min player
        else:
            min_score = float('inf')
            for move in self.get_moves(board_state, self.opponent_id):
                score = self.minimax(move, depth-1, True, alpha, beta)
                min_score = min(score, min_score)
                beta = min(beta, min_score)
                if beta <= alpha:
                    break

            return min_score

    def print_board(self, board_state):

        width = len(board_state)
        height = len(board_state[0])

        print("-----------------")


        # Traverse the rows
        for j in range(1, height+1):
            row_string = ""
            for i in range(width):
                row_string += str(board_state[i][-j]) + " "
            print(row_string)

    def calc_heuristic(self, board_state):
        """Calculates the estimated score of a given board. Positive
        scores means the board favors the ai, negative scores means
        the boad favors the opponent."""
        score = 0

        width = len(board_state)
        height = len(board_state[0])

        # Iterate through each space on the board
        for i, column in enumerate(board_state):
            for j, space_val in enumerate(board_state[i]):
                
                # Check rightward quartet
                if i+3 < width:
                    quartet = [board_state[i][j],
                               board_state[i+1][j],
                               board_state[i+2][j],
                               board_state[i+3][j]]

                    score += self.__calc_quartet_score(quartet)


                # Check upward quartet
                if j+3 < height:
                    quartet = [board_state[i][j],
                               board_state[i][j+1],
                               board_state[i][j+2],
                               board_state[i][j+3]]

                    score += self.__calc_quartet_score(quartet)

                # Check up-right quartet
                if i+3 < width and j+3 < height:
                    quartet = [board_state[i][j],
                               board_state[i+1][j+1],
                               board_state[i+2][j+2],
                               board_state[i+3][j+3]]

                    score += self.__calc_quartet_score(quartet)

                # Check down-right quartet
                if(i+3 < width and j-3 >= 0):
                    quartet = [board_state[i][j],
                               board_state[i+1][j-1],
                               board_state[i+2][j-2],
                               board_state[i+3][j-3]]

                    score += self.__calc_quartet_score(quartet)
        return score

    def __calc_quartet_score(self, quartet):
        """Calculates the score contribution of a given quartet."""

        # For each quartet check:
        # - contains at least one disc of each color = 0
        # - contains 4 discs of the same color = +/- inf
        # - contains 3 discs of the same color and one empty +/- 100
        # - contains 2 discs of the same color (2 empties) +/- 10
        # - contains 1 disc (and 3 empties) is worth +/- 1

        # Check at least one disc of each color
        if 1 in quartet and 2 in quartet:
            return 0

        # Check 4 discs of the same color
        if quartet[0] == quartet[1] == quartet[2] == quartet[3]:
            if quartet[0] == self.player_id:
                return float('inf')
            if quartet[0] != self.player_id and quartet[0] != 0:
                return -float('inf')

        sum_1 = sum( x == 1 for x in quartet)
        sum_2 = sum( x == 2 for x in quartet)

        if self.player_id == 1:
            if sum_1 == 4:
                return float('inf')
            if sum_1 == 3:
                return 100
            if sum_1 == 2:
                return 10
            if sum_1 == 1:
                return 1

            if sum_2 == 4:
                return -float('inf')
            if sum_2 == 3:
                return -100
            if sum_2 == 2:
                return -10
            if sum_2 == 1:
                return -1

        if self.player_id == 2:
            if sum_2 == 4:
                return float('inf')
            if sum_2 == 3:
                return 100
            if sum_2 == 2:
                return 10
            if sum_2 == 1:
                return 1

            if sum_1 == 4:
                return -float('inf')
            if sum_1 == 3:
                return -100
            if sum_1 == 2:
                return -10
            if sum_1 == 1:
                return -1

        return 0

    def get_moves(self, board_state, player):
        """Returns a list descibing the states of the board after
        making any legal move."""

        moves = []

        # Traverse the columns and make a move if the column has room.
        for i, col in enumerate(board_state):
            if col[-1] == 0:
                # The column is open for another move
                # Find the index of the first zero
                zero_index = col.index(0)

                # Make a copy of the current board state
                # and update the playable column
                new_board_state = [list(x) for x in board_state]
                new_board_state[i][zero_index] = player

                moves.append(new_board_state)

        return moves

    def get_moves(self, board_state, player):
        """Returns a list descibing the states of the board after
        making any legal move."""

        moves = []

        # Traverse the columns and make a move if the column has room.
        for i, col in enumerate(board_state):
            if col[-1] == 0:
                # The column is open for another move
                # Find the index of the first zero
                zero_index = col.index(0)

                # Make a copy of the current board state
                # and update the playable column
                new_board_state = [list(x) for x in board_state]
                new_board_state[i][zero_index] = player

                moves.append(new_board_state)

        return moves
if __name__ == '__main__':
    player = ComputerPlayer(1, 4)
    
    board = [[1, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 2, 0, 0, 0],
            [2, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1]]

    player.calc_heuristic(board)
