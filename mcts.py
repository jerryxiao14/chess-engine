import chess 
from chessEnv import ChessEnv
from node import Node
from edge import Edge
import config
import numpy as np
from tqdm import tqdm 
from chess import Board
from chess import PieceType
import logging

import cProfile
import pstats
import io


class MCTS:
    def __init__(self,agent, state = chess.STARTING_FEN,stochastic = False):
        self.root = Node(state=state)
        
        
        self.game_path = []
        self.cur_board = None 

        self.agent = agent 
        self.stochastic = stochastic

        self.expand(leaf=self.root)
        
    
    def run_simulations(self, n):
        for _ in tqdm(range(n)):
            self.game_path = []
            leaf = self.select_child(self.root)
            #logging.debug(f'selected leaf as the state:\n {Board(leaf.state)}')
            leaf.N +=1
            #logging.debug(f'leaf n value is now {leaf.N}')
            leaf = self.expand(leaf)
            leaf = self.back_propagate(leaf,leaf.value)
    
    def select_child(self, node):
        #Traverse the tree from node via selecting actions with max Q+U

        # And if node hasn't been visited, return the node

        #while node is not a leaf node
        
        #logging.debug(f'before selecting children')
        n = len(node.edges)
        noise = [1 for _ in range(n)]
        while node.N!=0:
            #logging.debug(f'checkign node with current state as:\n {Board(node.state)}')
            if len(node.edges)==0:
                logging.debug(f'current node has no outgoing edges')
                return node 
            
            best = None 
            best_score = -np.inf 
            
            n = len(node.edges)
            for i in range(n):
                edge = node.edges[i]
                #logging.debug(f'checking edge with state to node:\n {Board(edge.out_node.state)}')
                cur_score = edge.upper_confidence_bound(noise=noise[i])
                if cur_score>best_score:
                    best_score = cur_score
                    best = edge 
            
            node = best.out_node
            self.game_path.append(best)
        return node

    def get_index(self, piece_type, direction, distance):
        if piece_type == PieceType.KNIGHT:
            return 56+direction 
        return direction*8+distance 
    
    @staticmethod
    def get_underpromotion_move(piece_type, from_square, to_square):
        under_promotion_piece_type = -1
        if piece_type==PieceType.KNIGHT:
            under_promotion_piece_type=0
        elif piece_type==PieceType.BISHOP:
            under_promotion_piece_type = 1
        elif piece_type==PieceType.ROOK:
            under_promotion_piece_type=2
        else:
            raise Exception("Underpromotion piece not valid")
        
        diff = from_square-to_square 
        if to_square<8:
            direction = diff-8
        elif to_square>55:
            direction = diff+8
        return (under_promotion_piece_type,direction)
    
    @staticmethod
    def get_knight_move(from_square, to_square):
        diff = to_square-from_square
        if diff==15:
            return 0
            #return "NORTH_LEFT"
        if diff==17:
            return 1
            #return "NORTH_RIGHT"
        if diff==10:
            return 2
            #return "EAST_UP"
        if diff==-6:
            return 3
            #return "EAST_DOWN"
        if diff==-15:
            return 4
            #return "SOUTH_RIGHT"
        if diff==-17:
            return 5
            #return "SOUTH_LEFT"
        if diff==-10:
            return 6
            #return "WEST_DOWN"
        if diff==6:
            return 7
            #return "WEST_UP"
        raise Exception("invalid knight moves")

    @staticmethod
    def get_queen_like_move(from_square, to_square):
        diff = to_square-from_square 
        if diff%8==0:
            if diff>0:
                #direction = "NORTH"
                direction = 1
            else:
                #direction = "SOUTH"
                direction = 5
            dist = abs(diff)//8
        elif diff%9==0:
            if diff>0:
                #direction = "NORTHEAST"
                direction = 2
            else:
                #direction = "SOUTHWEST"
                direction = 6
            dist = abs(diff)//9
        elif from_square//8==to_square//8:
            if diff>0:
                #direction = "EAST"
                direction = 3
            else:
                #direction = "WEST"
                direction = 7
            dist = abs(diff)
        elif diff%7==0:
            if diff>0:
                #direction = "NORTHWEST"
                direction = 8
            else:
                #direction = "SOUTHEAST"
                direction = 4
            dist = abs(diff)//7
        else:
            raise Exception("NOT a valid queen-like move")
        return (direction,dist)

    def probabilities_to_actions(self,probabilities,board):
        probabilities=probabilities.reshape((config.amount_of_planes,config.n,config.n))
        actions = {}

        self.cur_board = chess.Board(board)
        valid_moves = self.cur_board.generate_legal_moves()
        self.outputs = []

        for move in valid_moves:
            from_square = move.from_square 
            to_square = move.to_square 
            plane_index = None 
            piece = self.cur_board.piece_at(from_square)
            direction = None 

            if move.promotion and move.promotion!=chess.QUEEN:
                piece_type, direction = self.get_underpromotion_move(move.promotion, from_square, to_square)
                plane_index = 64+ 3*piece_type + (1-direction)
            else:
                if piece.piece_type==chess.KNIGHT:
                    direction = self.get_knight_move(from_square,to_square)
                    plane_index = 56+direction 
                else:
                    direction, dist = self.get_queen_like_move(from_square,to_square)
                    plane_index = direction*7+dist 
            col = from_square % 8
            row = 7- from_square//8
            self.outputs.append((move,plane_index, row, col))
        
        for move, plane_index, row, col in self.outputs:
            actions[move.uci()] = probabilities[plane_index][row][col]
        return actions 

    def expand(self, leaf):
        profiler = cProfile.Profile()
        profiler.enable()

        # Original function code
        board = chess.Board(leaf.state)

        possible_actions = list(board.generate_legal_moves())
        if len(possible_actions) == 0:
            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                leaf.value = 0
            else:
                if outcome.winner == chess.WHITE:
                    leaf.value = 1
                else:
                    leaf.value = 0
            profiler.disable()

            # Save profiler results for this function
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats('cumulative')
            ps.print_stats()
            with open("expand_profiling.txt", "w") as f:
                f.write(s.getvalue())
            return leaf

        input = ChessEnv.state_to_input(leaf.state)
        p, v = self.agent.predict(input)

        actions = self.probabilities_to_actions(p, leaf.state)
        leaf.value = v

        for action in possible_actions:
            new_state = leaf.step(action)
            leaf.add_child(Node(new_state), action, actions[action.uci()])

        profiler.disable()

        # Save profiler results for this function
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats('cumulative')
        ps.print_stats()
        with open("expand_profiling.txt", "w") as f:
            f.write(s.getvalue())

        return leaf

    '''
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

        input = ChessEnv.state_to_input(leaf.state)
        p,v = self.agent.predict(input)

        actions = self.probabilities_to_actions(p,leaf.state)
        leaf.value = v 

        for action in possible_actions:
            new_state = leaf.step(action)
            leaf.add_child(Node(new_state),action, actions[action.uci()])
        return leaf 
    '''
    def back_propagate(self, end_node, value):
        for edge in self.game_path:
            edge.in_node.N+=1
            edge.N+=1
            edge.W+=value
        return end_node





