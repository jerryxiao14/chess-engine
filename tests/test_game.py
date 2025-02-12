from chess import *
import sys
import os
# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)


import game
import chessEnv 
import agent
import logging 

logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

#print("\n\n\n\n\n\n PATH IS ",os.path.dirname(__file__))
model_path = os.path.join(os.path.dirname(__file__),"..","models/base_model.keras")
#print(f'modelpath is {model_path}')
new_game = game.Game(chessEnv.ChessEnv(), agent.Agent(model_path = model_path), agent.Agent(model_path=model_path))
new_game.play_game()
