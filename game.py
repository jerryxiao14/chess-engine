import os 
from chessEnv import ChessEnv
from agent import Agent
import config
from edge import Edge 
from mcts import MCTS
import pandas as pd
import numpy as np
import chess
from chess.pgn import Game as ChessGame
import logging

# Configure logging
logging.basicConfig(
    filename='game_log.txt',  # file to write logs
    filemode='w',             # 'w' to overwrite each run, 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG        # log everything DEBUG and above
)

class Game:
    def __init__(self,env, white, black):
        self.env = env 
        self.white = white
        self.black = black 
        
        #Store memory of games played
        self.memory = []
    def reset(self):
        self.env.reset()
        self.turn = self.env.board.turn
    def play_game(self, stochastic = True):
        self.reset()
        self.memory.append([])

        move_counter = 0
        previous_edges = (None, None)

        winner = None
        
        while not self.env.board.is_game_over():
            previous_edges = self.play_move(stochastic = stochastic, previous_moves = previous_edges)
            move_counter+=1 
            logging.info(f'board after move {move_counter} is: \n{self.env.board}')

            if move_counter>350:
                winner = self.guess_winner()
                break
        
        if winner is None:
            game_result = self.env.board.result()
            if game_result == "1-0":
                winner = 1
            elif game_result=="0-1":
                winner = -1
            else:
                winner = 0
        
        #Copyting this part to see what it does

        game = ChessGame()
        # set starting position
        game.setup(self.env.fen)
        # add moves
        node = game.add_variation(self.env.board.move_stack[0])
        for move in self.env.board.move_stack[1:]:
            node = node.add_variation(move)
        # print pgn

        # save memory to file
        logging.info(game)
        #self.save_game(name="game", full_game=full_game)

        return winner
        
                
                
    
    def guess_winner(self):
        cur_score = 0
        piece_scores = {
            chess.PAWN: 1,
            chess.KNIGHT:3,
            chess.BISHOP:3,
            chess.ROOK:5,
            chess.QUEEN:9,
            chess.KING:20
        }

        for piece in self.env.board.piece_map.values():
            if piece.color== chess.WHITE:
                score+=piece_scores[piece.piece_type]
            else:
                score-=piece_scores[piece.piece_type]
        if score>2.5:
            return 1
        elif score<-2.5:
            return -1
        return 0

    def play_move(self, stochastic = True, previous_moves = (None,None),save_moves = True):
        current = self.white if self.turn else self.black
        logging.info(f'playing move as {current} with board as:\n {self.env.board}')
        if previous_moves[0] is None or previous_moves[1] is None:
            # Initialize mcts
            current.mcts = MCTS(current,state = self.env.board.fen(),stochastic = stochastic)
        else:
            try:
                node = current.mcts.root.get_edge(previous_moves[0].action).output_node
                node = node.get_edge(previous_moves[1].action).output_node
                current.mcts.root = node 
            except AttributeError:
                current.mcts = MCTS(current, state = self.env.board.fen(), stochastic = stochastic)
        
        logging.debug(f'before running simulations')
        current.mcts.run_simulations(300)
        logging.debug(f'after mcts running simulations')

        moves = current.mcts.root.edges 

        if save_moves:
            self.save_to_memory(self.env.board.fen(),moves)
        
        total_visits = sum(e.N for e in moves)
        probs = [e.N/ total_visits for e in moves]

        if stochastic:
            best = np.random.choice(moves,p = probs)
        else:
            best = moves[np.argmax(probs)]
        
        self.env.step(best.action)

        self.turn = not self.turn 

        return (previous_moves[1], best)

    def save_to_memory(self, state, moves):
        total_visits = sum(e.N for e in moves)

        probabilities = {
            e.action.uci(): e.N / total_visits for e in moves
        }

        self.memory[-1].append((state,probabilities,None))