# pit.py

import Arena
from MCTS import MCTS
from utils import *
import numpy as np

"""
Use this script to play any two agents against each other, or play manually with
any agent for the custom Tic-Tac-Toe game.
"""

# --- Configuration ---
# Set to True to use the PyTorch version of the NNet, False for Keras
USE_PYTORCH = True
# Set to True to play against the AI, False for AI vs. AI
HUMAN_VS_CPU = True

# --- Game and Player Imports ---
from tictacshoot.CustomTicTacToeGame import CustomTicTacToeGame as Game
from tictacshoot.CustomTicTacToePlayers import HumanPlayer, RandomPlayer

# --- Neural Network Import ---
if USE_PYTORCH:
    from tictacshoot.pytorch.NNet import NNetWrapper as NNet
else:
    from tictacshoot.keras.NNet import NNetWrapper as NNet

# --- Game Initialization ---
g = Game()

# --- Player Definitions ---
# Human player
hp = HumanPlayer(g).play
# Random player
rp = RandomPlayer(g).play

# --- Neural Network Player Setup ---
# Initialize the Neural Network
n1 = NNet(g)

# Define the checkpoint path and filename
if USE_PYTORCH:
    # NOTE: You must have a trained model for this to work.
    # To play against a random opponent, comment out n1.load_checkpoint and set player2 = rp
    # To train a model, run main.py.
    n1.load_checkpoint('./pretrained_models/tictacshoot/pytorch/', 'best.pth.tar')
else: # Keras
    n1.load_checkpoint('./pretrained_models/tictacshoot/keras/', 'best.h5')

# MCTS arguments
args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
# Create the MCTS agent
mcts1 = MCTS(g, n1, args1)
# Define the AI player function (takes the best action)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# --- Arena Setup ---
if HUMAN_VS_CPU:
    player2 = hp
else:
    player2 = n1p 


# The Arena will pit player n1p against player2
# The game's display function is used to print the board
# FIXED: Use instance method g.display instead of class method Game.display
arena = Arena.Arena(n1p, player2, g, display=g.display)

# Play 2 games and print the results
print(arena.playGames(2, verbose=True))