# tictactoe/CustomTicTacToeGame.py

from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .CustomTicTacToeLogic import Board
import numpy as np

class CustomTicTacToeGame(Game):
    """
    Game class implementation for the custom C++ Tic-Tac-Toe variant.
    It wraps the Board logic into the API required by the AlphaZero framework.
    """
    def __init__(self, n=3):
        self.n = n
        # 8 rotation options per square + 3 special actions
        self.action_size = n*n*8 + 3
        self.ACTION_SPIN = n*n*8
        self.ACTION_SHOOT = n*n*8 + 1
        self.ACTION_END_TURN = n*n*8 + 2

        # Arrow symbols matching the C++ source (Game.h)
        # C++ piece '1' (O) -> Python player '1'
        # C++ piece '2' (X) -> Python player '-1'
        self.symbols = {
            0: "⬜", # Empty
            1: ["\u21E8", "\u2B02", "\u21E9", "\u2B03", "\u21E6", "\u2B01", "\u21E7", "\u2B00"], # Player O
           -1: ["\u2192", "\u2198", "\u2193", "\u2199", "\u2190", "\u2196", "\u2191", "\u2197"]  # Player X
        }

    def getInitBoard(self):
        b = Board(self.n)
        return self._encode_board(b)

    def getBoardSize(self):
        # Board is encoded into planes; the framework typically uses (n, n).
        return (self.n, self.n)

    def getActionSize(self):
        return self.action_size

    def getNextState(self, board, player, action):
        b = self._decode_board(board)
        b.execute_move(action, player)
        next_player = -player if action == self.ACTION_END_TURN else player
        # self.display(board)
        return self._encode_board(b), next_player

    def getValidMoves(self, board, player):
        valids = [0] * self.getActionSize()
        b = self._decode_board(board)
        legal_moves = b.get_legal_moves(player)

        for move_idx in legal_moves:
            if 0 <= move_idx < self.ACTION_SPIN:
                valids[move_idx] = 1
            elif move_idx == self.ACTION_SPIN:
                valids[self.ACTION_SPIN] = 1
            elif move_idx == self.ACTION_SHOOT:
                valids[self.ACTION_SHOOT] = 1
            elif move_idx == self.ACTION_END_TURN:
                valids[self.ACTION_END_TURN] = 1

        return np.array(valids)

    def getGameEnded(self, board, player):
        b = self._decode_board(board)
        win_status = b.check_win()

        if win_status != 0:
            return win_status * player # 1 if current player won, -1 if they lost

        if b.turn_number > 500: 
            return 1e-4

        return 0 # Game not over

    def getCanonicalForm(self, board, player):
        canonical_board = np.copy(board)
        # Flip the piece plane so the current player is always '1'
        canonical_board[0, :, :] *= player 
        return canonical_board


    def getSymmetries(self, board, pi):
        # Symmetries are disabled due to the directional nature of rotations.
        # return [(board, pi)]  <-- OLD LINE
        
        # NEW LINE:
        return self.get_rotational_symmetries(board, pi, self.n)
    

    def get_rotational_symmetries(board, pi, n=3):
        """
        Generates rotational symmetries for a given board state and policy vector.

        Caveats:
        - Symmetries are NOT generated if the special token is active, as its
        position at (2,1) is not rotationally symmetric.
        - Only clockwise rotations (90, 180, 270 degrees) are generated. Reflections
        are excluded as game pieces only rotate clockwise.

        Args:
            board (np.array): The board state, with shape (7, n, n).
            pi (np.array): The policy vector of size (n*n*8 + 3).
            n (int): The dimension of the board (e.g., 3 for a 3x3 board).

        Returns:
            list: A list of (board, pi) tuples, including the original.
        """
        # Caveat: There are never rotations while the token is active.
        # The token's fixed starting position (2,1) breaks rotational symmetry.
        is_token_active = board[6, 0, 0]
        if is_token_active:
            return [(board, pi)]

        symmetries = []
        current_board, current_pi = np.copy(board), np.copy(pi)
        
        # The original orientation is always included.
        symmetries.append((current_board, current_pi))

        # Generate the 3 clockwise rotations (90, 180, 270 degrees).
        for i in range(3):
            # --- 1. Rotate the board state ---
            
            # The new board is a 90-degree clockwise rotation of the previous one.
            rotated_board = np.copy(symmetries[-1][0])
            
            # These planes contain spatial information and need to be rotated.
            # np.rot90(m, k=-1) performs a 90-degree clockwise rotation.
            for plane_idx in [0, 1, 2, 4]: # pieces, rotations, shields, last_placed
                rotated_board[plane_idx] = np.rot90(rotated_board[plane_idx], k=-1)
                
            # These planes contain scalar game-state info and are not rotated.
            # [3]: actions_left, [5]: turn_number, [6]: token_active

            # --- 2. Adjust the piece rotation values ---
            # When the board turns, the direction each piece faces also changes.
            # A 90-degree clockwise board rotation corresponds to adding 2 to the
            # rotation index (0-7), as each index step is 45 degrees.
            
            # Only update directions for squares that actually contain a piece.
            piece_mask = rotated_board[0] != 0
            
            # Add 2 to the rotation index, modulo 8 to wrap around.
            current_rotations = rotated_board[1][piece_mask]
            rotated_board[1][piece_mask] = (current_rotations + 2) % 8
            
            # --- 3. Rotate the policy vector ---
            
            # The previous policy vector is the one we are rotating.
            pi_to_rotate = symmetries[-1][1]
            rotated_pi = np.zeros_like(pi_to_rotate)
            
            action_size_placements = n * n * 8
            
            # The special actions (SPIN, SHOOT, END_TURN) are not spatial.
            rotated_pi[action_size_placements:] = pi_to_rotate[action_size_placements:]
            
            # Remap the placement probabilities.
            n_sq = n * n
            for r in range(n):
                for c in range(n):
                    for rot_idx in range(8):
                        # Original action's position and rotation
                        old_pos = r * n + c
                        old_action = rot_idx * n_sq + old_pos

                        # New position and rotation after a 90-degree CW turn
                        new_r, new_c = c, n - 1 - r # Coordinate transformation
                        new_pos = new_r * n + new_c
                        new_rot_idx = (rot_idx + 2) % 8 # Rotation value transformation
                        new_action = new_rot_idx * n_sq + new_pos
                        
                        # Move the probability to the new action index
                        rotated_pi[new_action] = pi_to_rotate[old_action]
            
            # Add the new symmetry to our list
            symmetries.append((rotated_board, rotated_pi))
            
        return symmetries


    def stringRepresentation(self, board):
        return board.tobytes()

    def display(self, board):
        b = self._decode_board(board)
        n = b.n

        player_char = "O" if b.turn_number % 2 == 0 else "X"
        print("-" * (6 * n))
        print(f"Turn: {b.turn_number} | Player: {player_char} | Actions Left: {b.actions_left} | Placed: {b.has_placed}")

        for r in range(n):
            if r > 0: print("-" * (6 * n))
            print(" | ", end="")
            for c in range(n):
                piece = b.pieces[r, c]

                if b.token_active and r == 2 and c == 1:
                     # Special display for the active C++ 'token'
                    symbol = " x "
                elif piece != 0:
                    rot = b.rotations[r, c]
                    symbol = self.symbols[piece][rot]
                    if b.has_shield_states[r,c]==1:
                        symbol="("+symbol+")"
                else:
                    symbol = self.symbols[0]

                print(f"{symbol:^3} | ", end="")
            print()
        print("-" * (6 * n))

    # --- Helper methods for encoding/decoding the board state ---

    def _encode_board(self, b):
        """ Encodes the Board object into a NumPy array for the NN. """
        # 7 planes: pieces, rotations, shields, actions_left, last_placed, turn_number, token_active
        board_state = np.zeros((7, self.n, self.n), dtype=np.float32)
        board_state[0] = b.pieces
        board_state[1] = b.rotations
        board_state[2] = b.has_shield_states
        board_state[3].fill(b.actions_left)
        if b.last_placed is not None:
            r, c = b.last_placed
            board_state[4, r, c] = 1.0
        board_state[5].fill(b.turn_number)
        board_state[6].fill(1 if b.token_active else 0)
        return board_state

    def _decode_board(self, board_state):
        """ Decodes the NumPy array back into a Board object. """
        b = Board(self.n)
        b.pieces = np.array(board_state[0], dtype=int)
        b.rotations = np.array(board_state[1], dtype=int)
        b.has_shield_states = np.array(board_state[2], dtype=int)
        b.actions_left = int(board_state[3, 0, 0])

        ys, xs = np.where(board_state[4] == 1)
        b.last_placed = (int(ys[0]), int(xs[0])) if len(ys) else None
        b.has_placed = b.last_placed is not None

        b.turn_number = int(board_state[5, 0, 0])
        b.token_active = bool(board_state[6, 0, 0])
        return b
