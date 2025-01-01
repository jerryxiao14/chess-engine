import chess 
from chessEnv import ChessEnv
from node import Node
from edge import Edge

import numpy as np
from tqdm import tqdm 
class MCTS:
    def __init__(self,agent, state = chess.STARTING_FEN,stochastic = False):
        self.root = Node(state=state)
        
        self.game_path = []
        self.cur_board = None 

        self.agent = agent 
        self.stochastic = stochastic
    
    def run_simultations(self, n):
        for _ in tqdm(range(n)):
            self.game_path = []
            leaf = self.select_child(self.root)
            leaf.N +=1
            leaf = self.expand(leaf)

            leaf = self.backpropagate(leaf,leaf.value)
    
    def select_child(self, node):
        #Traverse the tree from node via selecting actions with max Q+U

        # And if node hasn't been visited, return the node

        #while node is not a leaf node
        n = len(node.edges)
        noise = [1 for _ in range(n)]
        while not node.N!=0:
            if len(node.edges)==0:
                return node 
            
            best = None 
            best_score = -np.inf 
            
            for i in range(n):
                edge = node.edges[i]
                cur_score = edge.upper_confidence_bound(noise[i])
                if cur_score>best_score:
                    best_score = cur_score
                    best = edge 
            
            node = best.out_node
            self.game_path.append(best)
        return node
    

    def expand(self, leaf):
        # Add all move possibility to leaf node

        board = chess.Board(leaf.state)

        possible_actions = list(board.generate_legal_moves())
        if len(possible_actions)==0:
            outcome = board.outcome(claim_draw = True)
            if outcome is None:
                leaf.value = 0
            else:
                if outcome.winner == chess.WHITE:
                    leaf.value=1
                else:
                    leaf.value = 0 
            return leaf 

        # Otherwise we have possible actions to append



