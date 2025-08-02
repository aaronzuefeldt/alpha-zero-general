# tictactoe/CustomTicTacToeLogic.py

import numpy as np

class Board():
    """
    Manages the complex state of the custom Tic-Tac-Toe game,
    mirroring the C++ implementation's member variables and rules.
    """
    # Maps rotation index to a (row, col) direction vector
    DIRECTIONS = [
        (0, 1),   # 0: Right (corresponds to u8"\u2192")
        (1, 1),   # 1: Down-Right (corresponds to u8"\u2198")
        (1, 0),   # 2: Down (corresponds to u8"\u2193")
        (1, -1),  # 3: Down-Left (corresponds to u8"\u2199")
        (0, -1),  # 4: Left (corresponds to u8"\u2190")
        (-1, -1), # 5: Up-Left (corresponds to u8"\u2196")
        (-1, 0),  # 6: Up (corresponds to u8"\u2191")
        (-1, 1)   # 7: Up-Right (corresponds to u8"\u2197")
    ]

    def __init__(self, n=3):
        "Set up initial board configuration."
        self.n = n
        # --- Game State ---
        # 1 for Player O, -1 for Player X, 0 for empty
        self.pieces = np.zeros((self.n, self.n), dtype=int)
        
        # Player -1 ('X') starts with the special token at (2,1)
        # In C++, piece '2' is 'X'. Here, it's '-1'.
        self.pieces[2, 1] = -1 
        
        # Rotation index (0-7) for each piece
        self.rotations = np.zeros((self.n, self.n), dtype=int)
        
        # C++ starts at turn 2, we start at 0. Player 1 ('O') makes the first move.
        self.turn_number = 0
        
        # Number of actions left in the current turn
        self.actions_left = 2
        
        # True if a piece has been placed this turn
        self.has_placed = False
        
        # True if the special token is on the board
        self.token_active = True

    def get_legal_moves(self, player):
        """
        Returns a list of all legal moves, mirroring C++ execute_all_possible_actions.
        A move is an integer from 0 to 11.
        - 0-8: Place a piece at the corresponding board index.
        - 9: Spin
        - 10: Shoot
        - 11: End Turn
        """
        moves = []
        
        # 1. Check for PLACE actions (if no piece has been placed yet)
        if not self.has_placed:
            for r in range(self.n):
                for c in range(self.n):
                    if self.pieces[r, c] == 0:
                        moves.append(r * self.n + c)

        # 2. Check for actions that cost an action point
        if self.actions_left > 0:
            # SPIN is possible if there is at least one piece on the board
            # (excluding the single initial token if it's the only piece)
            if np.count_nonzero(self.pieces) > 1 or (np.count_nonzero(self.pieces) == 1 and not self.token_active):
                 moves.append(9) # SPIN
            
            # SHOOT is possible if there are valid targets
            if self._has_valid_targets(player):
                moves.append(10) # SHOOT

        # 3. END_TURN is always a valid move
        moves.append(11)

        return moves

    def _has_valid_targets(self, player):
        """Checks if a shoot action would have any effect."""
        for r_start in range(self.n):
            for c_start in range(self.n):
                if self.pieces[r_start, c_start] == player:
                    # The special token at (2,1) cannot be used to shoot while active.
                    # Your C++ code has a check: `(!token || i != 2 || j != 1)`
                    if self.token_active and r_start == 2 and c_start == 1:
                        continue
                    
                    rot_idx = self.rotations[r_start, c_start]
                    direction = self.DIRECTIONS[rot_idx]
                    r, c = r_start + direction[0], c_start + direction[1]
                    
                    while 0 <= r < self.n and 0 <= c < self.n:
                        if self.pieces[r, c] != 0:
                            return True # Found a valid target
                        r, c = r + direction[0], c + direction[1]
        return False

    def check_win(self):
        """
        Check whether a player has won.
        Returns 1 if Player O wins, -1 if Player X wins, 0 for no win.
        """
        for player in [1, -1]:
            # Check rows and columns
            for i in range(self.n):
                if np.all(self.pieces[i, :] == player) or np.all(self.pieces[:, i] == player):
                    return player
            # Check diagonals
            if np.all(np.diag(self.pieces) == player) or np.all(np.diag(np.fliplr(self.pieces)) == player):
                return player
        return 0

    def execute_move(self, move_idx, player):
        """Perform the given move on the board."""
        
        if 0 <= move_idx <= 8: # PLACE
            r, c = divmod(move_idx, self.n)
            assert self.pieces[r, c] == 0 and not self.has_placed
            self.pieces[r, c] = player
            self.rotations[r, c] = 0
            self.has_placed = True
        
        elif move_idx == 9: # SPIN
            assert self.actions_left > 0
            self.actions_left -= 1
            self.rotations = (self.rotations + 1) % 8

        elif move_idx == 10: # SHOOT
            assert self.actions_left > 0
            marked_for_removal = np.zeros((self.n, self.n), dtype=bool)
            for r_start in range(self.n):
                for c_start in range(self.n):
                    if self.pieces[r_start, c_start] == player:
                        if self.token_active and r_start == 2 and c_start == 1:
                            continue
                        
                        rot_idx = self.rotations[r_start, c_start]
                        direction = self.DIRECTIONS[rot_idx]
                        r, c = r_start + direction[0], c_start + direction[1]

                        while 0 <= r < self.n and 0 <= c < self.n:
                            if self.pieces[r, c] != 0:
                                marked_for_removal[r, c] = True
                                if r == 2 and c == 1:
                                    self.token_active = False
                                break
                            r, c = r + direction[0], c + direction[1]
            
            self.pieces[marked_for_removal] = 0
            self.rotations[marked_for_removal] = 0
            self.actions_left -= 1

        elif move_idx == 11: # END_TURN
            self.turn_number += 1
            self.actions_left = 2
            self.has_placed = False