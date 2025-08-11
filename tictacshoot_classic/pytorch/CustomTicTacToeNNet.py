# tictactoe/pytorch/CustomTicTacToeNNet.py

import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomTicTacToeNNet(nn.Module):
    def __init__(self, game, args):
        # --- Game Parameters ---
        self.input_shape = game.getInitBoard().shape # (5, 3, 3)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(CustomTicTacToeNNet, self).__init__()
        # Use input_shape[0] (e.g., 5) for the number of input channels
        self.conv1 = nn.Conv2d(self.input_shape[0], args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        # Add padding to conv3 and conv4 to handle the small 3x3 board size
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        # Recalculate the size of the fully connected layer's input
        # With padding, the spatial dimensions remain 3x3.
        fc_input_size = args.num_channels * self.board_x * self.board_y
        self.fc1 = nn.Linear(fc_input_size, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size) # Policy head
        self.fc4 = nn.Linear(512, 1) # Value head

    def forward(self, s):
        # s: batch_size x planes (5) x board_x x board_y
        # The input 's' is already in the correct format, no initial view/reshape needed.
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        
        # Flatten the output for the fully connected layers
        s = s.view(-1, self.args.num_channels * self.board_x * self.board_y)

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)