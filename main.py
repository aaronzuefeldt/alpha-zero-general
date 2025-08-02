# main.py

import logging

from Coach import Coach
from utils import *

# --- CHOOSE YOUR BACKEND AND GAME ---
# This should point to your PyTorch NNet wrapper
from tictacshoot.pytorch.NNet import NNetWrapper as nn 
# This should point to your custom game
from tictacshoot.CustomTicTacToeGame import CustomTicTacToeGame as Game


log = logging.getLogger(__name__)

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of MCTS simulations per move.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x8x25','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    log.info('Starting self-play...')

    g = Game()
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint.')

    c = Coach(g, nnet, args)
    if args.load_model:
        log.info("Loading trainExamples from file...")
        c.loadTrainExamples()
        
    c.learn()

if __name__ == "__main__":
    main()