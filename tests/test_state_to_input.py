from chess import *
import sys
import os

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import chessEnv

import unittest


fen = "8/8/8/8/8/8/k1K5/8 w - - 100 75"

board = chessEnv.ChessEnv(fen=fen)
print(board)

test_input = board.state_to_input(fen)
#print(f'test input is {test_input}')

        

