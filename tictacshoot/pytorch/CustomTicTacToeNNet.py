# CustomTicTacToe/pytorch/ResNetNNet.py  (Rename the file for clarity)

import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- The new Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

# --- The Redesigned Main Network ---
class ResNetNNet(nn.Module): # Renamed for clarity
    def __init__(self, game, args):
        super(ResNetNNet, self).__init__()
        # --- Game Parameters ---
        self.input_shape = game.getInitBoard().shape
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        
        # --- Neural Net Architecture ---

        # 1. Initial Convolutional "Stem"
        # This takes the raw input planes and creates the feature maps for the ResNet tower.
        self.conv_stem = nn.Conv2d(self.input_shape[0], args.num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(args.num_channels)

        # 2. Residual "Tower"
        # This is the body of the network. We stack N residual blocks.
        self.res_tower = nn.ModuleList([ResidualBlock(args.num_channels) for _ in range(args.num_residual_blocks)])

        # 3. Policy Head
        # Takes the final feature maps and outputs move probabilities.
        self.policy_conv = nn.Conv2d(args.num_channels, 2, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_x * self.board_y, self.action_size)

        # 4. Value Head
        # Takes the final feature maps and outputs a single value (-1 to 1).
        self.value_conv = nn.Conv2d(args.num_channels, 1, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * self.board_x * self.board_y, 256)
        self.value_fc2 = nn.Linear(256, 1)


    def forward(self, s):
        # Input 's' shape: batch_size x num_planes x board_x x board_y
        
        # Pass through the initial stem
        s = F.relu(self.bn_stem(self.conv_stem(s)))

        # Pass through the residual tower
        for block in self.res_tower:
            s = block(s)
        
        # --- Policy Head Path ---
        pi = F.relu(self.policy_bn(self.policy_conv(s)))
        pi = pi.view(-1, 2 * self.board_x * self.board_y)
        pi = self.policy_fc(pi)

        # --- Value Head Path ---
        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.view(-1, 1 * self.board_x * self.board_y)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)

        # Final activations
        return F.log_softmax(pi, dim=1), torch.tanh(v)