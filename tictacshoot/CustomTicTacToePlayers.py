# tictactoe/CustomTicTacToePlayers.py

import numpy as np

class RandomPlayer():
    """A player that chooses a legal move at random."""
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # Get a list of all valid moves
        valids = self.game.getValidMoves(board, 1)
        # Convert the binary vector to a list of action indices
        valid_moves = np.where(valids == 1)[0]
        # Choose one of the valid moves at random
        return np.random.choice(valid_moves)


class HumanPlayer():
    """A player that prompts a human for input."""
    def __init__(self, game):
        self.game = game
        self.action_size = game.getActionSize()
        self.ACTION_SPIN = game.n * game.n
        self.ACTION_SHOOT = game.n * game.n + 1
        self.ACTION_END_TURN = game.n * game.n + 2

    def play(self, board):
        # Get a list of all valid moves
        valid_moves = self.game.getValidMoves(board, 1)
        
        print("\n--- Available Actions ---")
        # Display all valid moves to the human player
        for i, is_valid in enumerate(valid_moves):
            if is_valid:
                if 0 <= i < self.ACTION_SPIN:
                    r, c = divmod(i, self.game.n)
                    print(f"[{i}] Place piece at ({r}, {c})")
                elif i == self.ACTION_SPIN:
                    print(f"[{i}] Spin all pieces")
                elif i == self.ACTION_SHOOT:
                    print(f"[{i}] Shoot")
                elif i == self.ACTION_END_TURN:
                    print(f"[{i}] End Turn")
        
        while True:
            try:
                # Get user input
                a = int(input("Enter the number of your desired action: "))
                # Check if the chosen action is in the list of valid moves
                if valid_moves[a]:
                    return a
                else:
                    print("Invalid action. Please choose from the list.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid action number.")