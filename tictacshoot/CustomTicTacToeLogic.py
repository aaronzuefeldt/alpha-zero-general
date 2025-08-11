# tictactoe/CustomTicTacToeLogic_sliding.py

import numpy as np

class Board():
    """
    Manages the custom Tic-Tac-Toe with shooting + sliding.
    Action space (for n=3): 72 placement actions (8 rotations × 9 squares)
    plus 3 special actions: 72=SPIN, 73=SHOOT, 74=END_TURN.
    """
    # Maps rotation index to a (row, col) direction vector
    DIRECTIONS = [
        (0, 1),   # 0: Right (→)
        (1, 1),   # 1: Down-Right (↘)
        (1, 0),   # 2: Down (↓)
        (1, -1),  # 3: Down-Left (↙)
        (0, -1),  # 4: Left (←)
        (-1, -1), # 5: Up-Left (↖)
        (-1, 0),  # 6: Up (↑)
        (-1, 1)   # 7: Up-Right (↗)
    ]

    def __init__(self, n=3):
        "Set up initial board configuration."
        self.n = n
        # 1 for Player O, -1 for Player X, 0 for empty
        self.pieces = np.zeros((self.n, self.n), dtype=int)

        # 1 = has shield, 0 = no shield
        self.has_shield_states = np.zeros((self.n, self.n), dtype=int)

        # Player -1 ('X') starts with the special token at (2,1) with NO shield
        self.pieces[2, 1] = -1 

        # Rotation index (0-7) for each piece
        self.rotations = np.zeros((self.n, self.n), dtype=int)

        self.turn_number = 0
        self.actions_left = 2
        self.has_placed = False
        self.last_placed = None
        self.token_active = True

    # ---------------------- Helpers ----------------------
    def _in_bounds(self, r, c):
        return 0 <= r < self.n and 0 <= c < self.n

    def _rc_to_idx(self, r, c):
        return r * self.n + c

    def _idx_to_rc(self, idx):
        return divmod(idx, self.n)

    def _rotate_dir_idx_for_slide(self, dir_idx, attempt):
        """
        attempt: 1,2,3  -> rotate +90°, +180°, +90° respectively.
        Using 8-direction ring, +2 steps = +90°, +4 = +180°.
        """
        if attempt == 2:
            return (dir_idx + 4) % 8
        else: # 1 or 3
            return (dir_idx + 2) % 8

    # ---------------------- Rules / API ----------------------
    def get_legal_moves(self, player):
        """
        Returns a list of all legal moves.
        A move is an integer:
        - 0..(8*n*n - 1): Place a piece at square (r,c) with rotation p in [0..7].
                           Encoding: move = p*(n*n) + r*n + c
        - 8*n*n: SPIN
        - 8*n*n + 1: SHOOT
        - 8*n*n + 2: END_TURN
        """
        moves = []

        PLACEMENT_BASE = 0
        SPECIAL_BASE = 8 * self.n * self.n

        # 1. PLACE
        if not self.has_placed:
            for p in range(8):
                for r in range(self.n):
                    for c in range(self.n):
                        if self.pieces[r, c] == 0:
                            moves.append(PLACEMENT_BASE + p*(self.n*self.n) + (r * self.n + c))

        # 2. Actions that cost an action point
        if self.actions_left > 0:
            # SPIN possible if there's at least one piece (and token can't be the only piece that blocks this)
            if np.count_nonzero(self.pieces) > 1 or (np.count_nonzero(self.pieces) == 1 and not self.token_active):
                 moves.append(SPECIAL_BASE) # SPIN

            # SHOOT possible if there are valid targets
            if self._has_valid_targets(player):
                moves.append(SPECIAL_BASE + 1) # SHOOT

        # 3. END_TURN
        if self.has_placed or np.count_nonzero(self.pieces) == self.n * self.n:
            moves.append(SPECIAL_BASE + 2) # END TURN

        return moves

    def _has_valid_targets(self, player):
        """Checks if a shoot action would have any effect."""
        for r_start in range(self.n):
            for c_start in range(self.n):
                if self.pieces[r_start, c_start] == player:
                    # Special token at (2,1) cannot shoot while active
                    if self.token_active and r_start == 2 and c_start == 1:
                        continue

                    rot_idx = self.rotations[r_start, c_start]
                    dr, dc = self.DIRECTIONS[rot_idx]
                    r, c = r_start + dr, c_start + dc

                    while self._in_bounds(r, c):
                        if self.pieces[r, c] != 0:
                            return True
                        r, c = r + dr, c + dc
        return False

    def check_win(self):
        """
        Check whether a player has won.
        Returns 1 if Player O wins, -1 if Player X wins, 0 for no win.
        """
        for player in [1, -1]:
            # Rows/cols
            for i in range(self.n):
                if np.all(self.pieces[i, :] == player) or np.all(self.pieces[:, i] == player):
                    return player
            # Diagonals
            if np.all(np.diag(self.pieces) == player) or np.all(np.diag(np.fliplr(self.pieces)) == player):
                return player
        return 0

    def execute_move(self, move_idx, player):
        """Perform the given move on the board."""

        SPECIAL_BASE = 8 * self.n * self.n

        if 0 <= move_idx < SPECIAL_BASE: # PLACE
            p, mod = divmod(move_idx, self.n * self.n)
            r, c = divmod(mod, self.n)
            assert self.pieces[r, c] == 0 and not self.has_placed
            self.pieces[r, c] = player
            self.has_shield_states[r, c] = 1  # new piece gets a shield
            self.rotations[r, c] = p  # 0..7
            self.has_placed = True
            self.last_placed = (r, c)

        elif move_idx == SPECIAL_BASE: # SPIN
            assert self.actions_left > 0
            self.actions_left -= 1
            self.rotations = (self.rotations + 1) % 8

        elif move_idx == SPECIAL_BASE + 1: # SHOOT
            assert self.actions_left > 0

            # 1) Gather all hits (first target in each ray)
            hits = {}  # (r,c) -> dir_idx that hit it (first one encountered is enough)
            for r_start in range(self.n):
                for c_start in range(self.n):
                    if self.pieces[r_start, c_start] == player:
                        # Special token cannot shoot while active
                        if self.token_active and r_start == 2 and c_start == 1:
                            continue
                        dir_idx = int(self.rotations[r_start, c_start])
                        dr, dc = self.DIRECTIONS[dir_idx]
                        r, c = r_start + dr, c_start + dc
                        while self._in_bounds(r, c):
                            if self.pieces[r, c] != 0:
                                hits.setdefault((r, c), dir_idx)  # keep first shooter direction
                                break
                            r, c = r + dr, c + dc

            if not hits:
                # No effect; still consumes an action point
                self.actions_left -= 1
                return

            # 2) Partition into removals vs slides based on shields
            will_die = set()
            will_slide = {}  # (r,c) -> dir_idx
            for (r, c), dir_idx in hits.items():
                if self.has_shield_states[r, c] == 0:
                    will_die.add((r, c))
                else:
                    will_slide[(r, c)] = dir_idx

            # Token deactivation if it gets hit and dies
            if (2, 1) in will_die:
                self.token_active = False

            # 3) Plan sliding destinations (treat dying pieces as empty when planning)
            # slide_targets: dest_idx -> list of [origin_idx, prev_idx, distance]
            slide_targets = {}
            occ = (self.pieces != 0).copy()
            for (r,c) in will_die:
                occ[r, c] = False  # consider them empty for slide planning

            def plan_slide_from(rc, dir_idx):
                r0, c0 = rc
                origin_idx = self._rc_to_idx(r0, c0)
                cur_dir_idx = dir_idx
                for attempt in (1,2,3):
                    j = 1
                    dr, dc = self.DIRECTIONS[cur_dir_idx]
                    while True:
                        rr = r0 + dr * j
                        cc = c0 + dc * j
                        # stop when out of bounds or blocked (occupied in occ)
                        if not self._in_bounds(rr, cc) or occ[rr, cc]:
                            if j > 1:
                                rr2 = r0 + dr * (j - 1)
                                cc2 = c0 + dc * (j - 1)
                                rr3 = r0 + dr * (j - 2)
                                cc3 = c0 + dc * (j - 2)
                                dest_idx = self._rc_to_idx(rr2, cc2)
                                prev_idx = self._rc_to_idx(rr3, cc3)
                                slide_targets.setdefault(dest_idx, []).append([origin_idx, prev_idx, j])
                                return dest_idx
                            else:
                                # immediate block -> rotate and retry
                                cur_dir_idx = self._rotate_dir_idx_for_slide(cur_dir_idx, attempt)
                                break  # break inner while to change direction
                        else:
                            j += 1
                # No valid slide; stays in place but shield is removed
                return origin_idx

            for rc, dir_idx in will_slide.items():
                plan_slide_from(rc, dir_idx)

            # 4) Resolve overlapping slide destinations
            #    Loop until every destination has at most one contender
            while True:
                overlaps = 0
                for dest_idx, contenders in list(slide_targets.items()):
                    if len(contenders) > 1:
                        overlaps += 1
                        # sort by distance (ascending)
                        contenders.sort(key=lambda x: x[2])
                        # compute index "step" from prev -> dest for backoff math
                        # Using linear indices is safe because step is constant along a ray: step = dest - prev
                        step = dest_idx - contenders[0][1]  # (dr*n + dc)
                        if len(contenders) >= 2 and contenders[0][2] == contenders[1][2]:
                            # tie: push everyone back one cell
                            for origin_idx, prev_idx, dist in contenders:
                                new_dest = prev_idx
                                new_prev = prev_idx - step  # 2*prev - dest
                                slide_targets.setdefault(new_dest, []).append([origin_idx, new_prev, dist - 1])
                            slide_targets[dest_idx] = []
                        else:
                            # winner stays; everyone else backs off one
                            winner = contenders[0]
                            losers = contenders[1:]
                            slide_targets[dest_idx] = [winner]
                            for origin_idx, prev_idx, dist in losers:
                                new_dest = prev_idx
                                new_prev = prev_idx - step
                                slide_targets.setdefault(new_dest, []).append([origin_idx, new_prev, dist - 1])
                if overlaps == 0:
                    break  # conflict-free

            # 5) Apply removals
            for (r, c) in will_die:
                self.pieces[r, c] = 0
                self.rotations[r, c] = 0
                self.has_shield_states[r, c] = 0

            # 6) Apply slides
            for dest_idx, contenders in slide_targets.items():
                if len(contenders) == 1:
                    origin_idx, prev_idx, dist = contenders[0]
                    if dest_idx != origin_idx:
                        orr, occc = self._idx_to_rc(origin_idx)
                        drr, dcc = self._idx_to_rc(dest_idx)
                        # move the piece
                        self.pieces[drr, dcc] = self.pieces[orr, occc]
                        self.rotations[drr, dcc] = self.rotations[orr, occc]
                        self.has_shield_states[drr, dcc] = 0  # shield consumed
                        # clear origin
                        self.pieces[orr, occc] = 0
                        self.rotations[orr, occc] = 0
                        self.has_shield_states[orr, occc] = 0
                    else:
                        # no move, but shield still consumed
                        r0, c0 = self._idx_to_rc(origin_idx)
                        self.has_shield_states[r0, c0] = 0

            self.actions_left -= 1

        else: # END_TURN
            self.turn_number += 1
            self.actions_left = 2
            self.has_placed = False
            self.last_placed = None
        

