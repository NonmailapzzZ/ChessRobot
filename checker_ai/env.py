from config import BOARD_SIZE, EMPTY, AI, OPP
from utils import move_to_action, action_to_move
import torch


class CheckersEnv:
    def __init__(self):
        self.board_size = BOARD_SIZE
        self.action_size = self.board_size ** 4
        self.reset()

    # ---------- utilities ----------
    def in_bounds(self, r, c):
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def is_black_square(self, r, c):
        return (r + c) % 2 == 1

    def count_pieces(self, player):
        return sum(
            1
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if self.board[r][c] == player
        )

    def get_state_tensor(self):
        return torch.tensor(self.board, dtype=torch.float32).unsqueeze(0)

    def get_state(self):
        return self.get_state_tensor()

    # ---------- reset ----------
    def reset(self):
        self.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]

        # ฝั่งตรงข้าม (บน)
        for r in range(2):
            for c in range(BOARD_SIZE):
                if self.is_black_square(r, c):
                    self.board[r][c] = OPP

        # ฝั่งเรา (ล่าง)
        for r in range(BOARD_SIZE - 2, BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.is_black_square(r, c):
                    self.board[r][c] = AI

        self.current_player = AI
        return self.get_state()

    # ---------- moves ----------
    def valid_moves(self):
        normal_moves = []
        capture_moves = []

        direction = -1 if self.current_player == AI else 1
        opponent = OPP if self.current_player == AI else AI

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] != self.current_player:
                    continue

                # ----- capture -----
                for dc in (-1, 1):
                    mr = r + direction
                    mc = c + dc
                    er = r + 2 * direction
                    ec = c + 2 * dc

                    if (
                        self.in_bounds(er, ec)
                        and self.is_black_square(er, ec)
                        and self.board[mr][mc] == opponent
                        and self.board[er][ec] == EMPTY
                    ):
                        capture_moves.append(((r, c), (er, ec)))

                # ----- normal move -----
                for dc in (-1, 1):
                    nr = r + direction
                    nc = c + dc
                    if (
                        self.in_bounds(nr, nc)
                        and self.is_black_square(nr, nc)
                        and self.board[nr][nc] == EMPTY
                    ):
                        normal_moves.append(((r, c), (nr, nc)))

        return capture_moves if capture_moves else normal_moves

    def valid_actions(self):
        return [move_to_action(m) for m in self.valid_moves()]

    # ---------- capture helpers ----------
    def can_capture_from(self, x, y, piece):
        opponent = OPP if piece == AI else AI
        directions = [(-2, -2), (-2, 2), (2, -2), (2, 2)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            mx, my = x + dx // 2, y + dy // 2

            if self.in_bounds(nx, ny):
                if self.board[mx][my] == opponent and self.board[nx][ny] == EMPTY:
                    return True
        return False

    def get_next_capture(self, x, y, piece):
        opponent = OPP if piece == AI else AI
        directions = [(-2, -2), (-2, 2), (2, -2), (2, 2)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            mx, my = x + dx // 2, y + dy // 2

            if self.in_bounds(nx, ny):
                if self.board[mx][my] == opponent and self.board[nx][ny] == EMPTY:
                    return nx, ny, mx, my
        return None

    def is_vulnerable(self, x, y, piece):
        opponent = OPP if piece == AI else AI
        directions = [(-2, -2), (-2, 2), (2, -2), (2, 2)]

        for dx, dy in directions:
            ox, oy = x + dx // 2, y + dy // 2
            ex, ey = x - dx, y - dy

            if self.in_bounds(ex, ey) and self.in_bounds(ox, oy):
                if self.board[ox][oy] == opponent and self.board[ex][ey] == EMPTY:
                    return True
        return False

    # ---------- step ----------
    def step(self, action):
        reward = 0.0
        done = False

        from_pos, to_pos = action_to_move(action)
        fx, fy = from_pos
        tx, ty = to_pos

        piece = self.board[fx][fy]
        opponent = OPP if piece == AI else AI

        # move
        self.board[fx][fy] = EMPTY
        self.board[tx][ty] = piece
        reward -= 0.05

        # capture
        dx = tx - fx
        dy = ty - fy

        if abs(dx) == 2 and abs(dy) == 2:
            mx = fx + dx // 2
            my = fy + dy // 2

            if self.board[mx][my] == opponent:
                self.board[mx][my] = EMPTY
                reward += 1.0

                # multi-capture
                while self.can_capture_from(tx, ty, piece):
                    nxt = self.get_next_capture(tx, ty, piece)
                    if nxt is None:
                        break
                    nx, ny, cx, cy = nxt
                    self.board[tx][ty] = EMPTY
                    self.board[cx][cy] = EMPTY
                    self.board[nx][ny] = piece
                    tx, ty = nx, ny
                    reward += 0.5

        # vulnerability penalty
        if self.is_vulnerable(tx, ty, piece):
            reward -= 0.3

        # terminal
        if self.count_pieces(opponent) == 0:
            reward += 10.0
            done = True
        elif self.count_pieces(piece) == 0:
            reward -= 10.0
            done = True

        self.current_player = opponent

        # ไม่มีตาเดิน = แพ้
        if len(self.valid_moves()) == 0:
            reward -= 10.0
            done = True

        return self.get_state(), reward, done, {}