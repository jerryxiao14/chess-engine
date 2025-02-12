import cProfile
import pstats

from chess import *
import sys
import os
# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import game
import chessEnv 
import agent
from mcts import MCTS
import logging   # Import your MCTS class and other dependencies

base_model_path = os.path.join(os.path.dirname(__file__), "..", "models/base_model.keras")

def profile_simulations():
    agent_lol = agent.Agent(base_model_path)  # Initialize or mock your agent
    mcts = MCTS(agent_lol)
    mcts.run_simulations(10)  # Test with a small number of simulations

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    profile_simulations()
    profiler.disable()

    # Save the profiling stats to a file
    output_file = "profiling_results.txt"
    with open(output_file, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats('cumulative')  # Sort by cumulative time
        stats.print_stats()  # Write all results to the file

    print(f"Profiling results written to {output_file}")
