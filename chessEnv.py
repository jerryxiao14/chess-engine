import chess
import numpy as np
from chess import Move, Board


class ChessEnv:
    def __init__(self, fen: str = chess.STARTING_FEN):
        self.fen = fen
        self.board = Board(self.fen)


    def step(self, move: Move) -> Board:
        self.board.push(move)
        return self.board


    def __str__(self):
        return str(self.board)

