import numpy as np

class CheckersEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((8,8), dtype=np.int8)

        for r in range(3):
            for c in range(8):
                if (r+c) % 2 == 1:
                    self.board[r][c] = -1

        for r in range(5,8):
            for c in range(8):
                if (r+c) % 2 == 1:
                    self.board[r][c] = 1

        return self.board.copy()

    def valid_moves(self):
        moves = []
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == 1:
                    for dr, dc in [(-1,-1), (-1,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < 8 and 0 <= nc < 8:
                            if self.board[nr][nc] == 0:
                                moves.append(((r,c),(nr,nc)))
        return moves

    def step(self, move):
        (fr,fc),(tr,tc) = move

        if ((fr,fc),(tr,tc)) not in self.valid_moves():
            return self.board, -1, False

        self.board[fr][fc] = 0
        self.board[tr][tc] = 1

        reward = 0.1
        done = np.sum(self.board == -1) == 0
        if done:
            reward = 10

        return self.board.copy(), reward, done
