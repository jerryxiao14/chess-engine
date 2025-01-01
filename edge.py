
#from node import Node
from chess import Move, Board
import chess

import math 
import numpy as np 
class Edge:
    def __init__(self, in_node, out_node, action, prior):
        self.in_node = in_node
        self.out_node = out_node
        self.action = action 

        self.player_turn = self.in_node.state.split(" ")[1]=="w" 

        # We have four parameters for each edge struct:
        # N: number of times this action has been used from the in_node state
        # W: Action value in total
        # P: Prior probability of selecting this action

        self.N = 0
        self.W = 0
        self.P = prior

    def upper_confidence_bound(self, noise:float) ->float:
        c = math.sqrt(2)

        exploitative_term = self.W/self.N 
        exploratory_term = c*(self.P*noise)*math.sqrt(math.log(self.in_node.N)/self.N)
        if self.in_node.turn==chess.WHITE:
            return exploitative_term+exploratory_term
        return -1*exploitative_term+exploratory_term
    
    
