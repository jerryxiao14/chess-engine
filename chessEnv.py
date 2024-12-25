import chess
import numpy as np
from chess import Move


class ChessEnv:
    def __init__(self, fen: str = chess.STARTING_FEN):
        self.fen = fen
        self.board = chess.Board(self.fen)





    def __str__(self):
        return str(self.board)

