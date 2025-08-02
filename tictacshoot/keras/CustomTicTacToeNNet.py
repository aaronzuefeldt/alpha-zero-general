# tictacshoot/keras/CustomTicTacToeNNet.py

import sys
sys.path.append('..')
from utils import *

import argparse
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

class CustomTicTacToeNNet():
    def __init__(self, game, args):
        # --- Game Parameters ---
        # The input shape is now 3D: (planes, board_x, board_y)
        self.input_shape = game.getInitBoard().shape 
        self.action_size = game.getActionSize()
        self.args = args

        # --- Neural Net ---
        # Input layer now expects a 3D board state (e.g., (5, 3, 3))
        self.input_boards = Input(shape=self.input_shape)

        # The input already has a 'channel' dimension (the 5 planes).
        # We specify 'channels_first' to the Conv2D layers.
        # The axis for BatchNormalization is 1 (the channel axis) for Conv2D layers.
        x_image = self.input_boards
        h_conv1 = Activation('relu')(BatchNormalization(axis=1)(Conv2D(args.num_channels, 3, padding='same', data_format='channels_first')(x_image)))
        h_conv2 = Activation('relu')(BatchNormalization(axis=1)(Conv2D(args.num_channels, 3, padding='same', data_format='channels_first')(h_conv1)))
        h_conv3 = Activation('relu')(BatchNormalization(axis=1)(Conv2D(args.num_channels, 3, padding='same', data_format='channels_first')(h_conv2)))
        h_conv4 = Activation('relu')(BatchNormalization(axis=1)(Conv2D(args.num_channels, 3, padding='valid', data_format='channels_first')(h_conv3)))
        
        h_conv4_flat = Flatten()(h_conv4)
        # For Dense layers, BatchNormalization axis should be -1 (or 1 for the feature axis)
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization()(Dense(1024)(h_conv4_flat))))
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization()(Dense(512)(s_fc1))))
        
        # Output Heads
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))