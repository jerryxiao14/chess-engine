from chess import *
import sys
import os
# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)


import game
import chessEnv 
import agent
import mcts 
import logging 

logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
base_model_path = os.path.join(os.path.dirname(__file__),"..","models/base_model.keras")
test_mcts = mcts.MCTS(agent=agent.Agent(model_path = base_model_path))

test_mcts.run_simulations(10)