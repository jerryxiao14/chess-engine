import os 
from chessEnv import ChessEnv
from agent import Agent
import config
from edge import Edge 
from mcts import MCTS
import pandas as pd
import numpy as np

class Game:
    def __init__(self,env, white, black):
        self.env = env 
        self.white = white
        self.black = black 
    
    