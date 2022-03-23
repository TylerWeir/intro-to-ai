"""
This Connect Four player using the minimax algorithm optimized with
alpha-beta pruning to dominate all human adversaries, including its
creator.
"""
__author__ = "Tyler Weir"
__license__ = "MIT"
__date__ = "March 3, 2022"

class ComputerPlayer:
    """
    This class represents a computer player in the Connect Four
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
        
        # Used for fast quartet score lookup
        self.quartet_scores = precompute_quartets()

    def test(self, rack):
        print(self.calc_heuristic(rack))

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

        # Vars to keep track of best move
        best_move = None
        best_move_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
    
        print("Scores")
        # Use minimax to score the possbible moves
        # This first layer is a max player
        for move in self.get_moves(rack, self.player_id):
            score = self.minimax(move, self.difficulty, False, alpha, beta)
            print(score)

            if score > best_move_score:
                best_move_score = score
                best_move = move

            alpha = max(alpha, best_move_score)
            if beta <= alpha:
                break

        # This means the bot will lose
        if best_move == None:
            # Just take the first move
            best_move = self.get_moves(rack, self.player_id)[0]

        # Decode the move from the board state
        for i, col in enumerate(rack):
            num_spaces = sum(x == 0 for x in col)
            num_space_move = sum(x == 0 for x in best_move[i])

            if num_spaces != num_space_move:
                #This is where the tile was played
                return i

    def minimax(self, board_state, depth, max_player, alpha, beta):
        """Uses the minimax algorthm to score a board state. Returns
        the score of the board state along with the final alpha and 
        beta values."""

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

    def calc_heuristic(self, board_state):
        """Calculates the estimated score of a given board. Positive
        scores means the board favors the ai, negative scores means
        the boad favors the opponent."""
        score = 0

        width = len(board_state)
        height = len(board_state[0])

        # Iterate through each space on the board
        for i in range(width):
            for j in range(height):

                # Check rightward quartet
                if i+3 < width:
                    quartet = (board_state[i][j],
                               board_state[i+1][j],
                               board_state[i+2][j],
                               board_state[i+3][j])

                    score += self.quartet_scores.get(quartet)[self.player_id-1]

                # Check upward quartet
                if j+3 < height:
                    quartet = (board_state[i][j],
                               board_state[i][j+1],
                               board_state[i][j+2],
                               board_state[i][j+3])

                    score += self.quartet_scores.get(quartet)[self.player_id-1]

                # Check up-right quartet
                if i+3 < width and j+3 < height:
                    quartet = (board_state[i][j],
                               board_state[i+1][j+1],
                               board_state[i+2][j+2],
                               board_state[i+3][j+3])

                    score += self.quartet_scores.get(quartet)[self.player_id-1]

                # Check down-right quartet
                if(i+3 < width and j-3 >= 0):
                    quartet = (board_state[i][j],
                               board_state[i+1][j-1],
                               board_state[i+2][j-2],
                               board_state[i+3][j-3])

                    score += self.quartet_scores.get(quartet)[self.player_id-1]

        return score


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

                # Add the board to the list in a tuple to be
                # sorted according to the player requesting the
                # moves. The i value is included to break score
                # ties.
                board_score = self.calc_heuristic(new_board_state)
                moves.append((board_score, i, new_board_state))

        # Min player wants lowest score first (accending order)
        # Max player wants highest score first (decending order)
        max_player = player == self.player_id
        moves.sort(reverse=max_player)

        # extract the boards and return
        return [c for (a, b, c) in moves]

def precompute_quartets():
    """Computes the scores of all quartets and returns a dictionary 
    mapping the quartet to a tuple where the first index is the 
    score for player 1 and the second index is the score for player 2."""

    # Make list of all possible quartets
    tiles = [0, 1, 2]
    all_quartets = [(i, j, k, l) for i in tiles
                                 for j in tiles
                                 for k in tiles 
                                 for l in tiles]

    # Score each quartet and store scores in dict
    score_dict = {}
    for quartet in all_quartets:
        score_dict.update({quartet: calc_quartet_score(quartet)})

    return score_dict

def calc_quartet_score(quartet):
    """Calculates the score contribution of a given quartet. Returns
    a tuple where the first element is player 1 score for the quartet 
    and the second element is player 2 score for the quartet."""
    # For each quartet check:
    # - contains at least one disc of each color = 0
    # - contains 4 discs of the same color = +/- inf
    # - contains 3 discs of the same color and one empty +/- 100
    # - contains 2 discs of the same color (2 empties) +/- 10
    # - contains 1 disc (and 3 empties) is worth +/- 1

    # Check at least one disc of each color
    if 1 in quartet and 2 in quartet:
        return (0, 0)
    
    # Calculate the score for each player
    player1_score = 0
    player2_score = 0

    sum_1 = sum( x == 1 for x in quartet)
    sum_2 = sum( x == 2 for x in quartet)

    # Calc player1 score
    if sum_1 == 4:
        player1_score = float('inf')
    if sum_1 == 3:
        player1_score = 100
    if sum_1 == 2:
        player1_score = 10
    if sum_1 == 1:
        player1_score = 1

    if sum_2 == 4:
        player1_score = -float('inf')
    if sum_2 == 3:
        player1_score = -100
    if sum_2 == 2:
        player1_score = -10
    if sum_2 == 1:
        player1_score = -1

    # Calc player2 score
    if sum_2 == 4:
        player2_score = float('inf')
    if sum_2 == 3:
        player2_score = 100
    if sum_2 == 2:
        player2_score = 10
    if sum_2 == 1:
        player2_score = 1

    if sum_1 == 4:
        player2_score = -float('inf')
    if sum_1 == 3:
        player2_score = -100
    if sum_1 == 2:
        player2_score = -10
    if sum_1 == 1:
        player2_score = -1

    return (player1_score, player2_score)


if __name__ == "__main__":
    board = [[0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 2, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0]]

    computer = ComputerPlayer(1, 1)
    computer.test(board)
