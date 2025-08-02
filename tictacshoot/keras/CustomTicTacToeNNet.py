# tictactoe/keras/CustomTicTacToeNNet.py

import sys
sys.path.append('..')
from utils import *

import argparse
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

class CustomTicTacToeNNet():
    def __init__(self, game, args):
        # Game Parameters
        self.input_shape = game.getInitBoard().shape 
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=self.input_shape)    # s: batch_size x planes x board_x x board_y

        # --- CHANGE FOR CPU COMPATIBILITY ---
        # Reshape input from (planes, x, y) to (x, y, planes) for NHWC format
        # The Permute layer reorders the dimensions. (1,2,3) -> (2,3,1)
        x_image = Permute((2, 3, 1))(self.input_boards)

        # The 'data_format' argument is removed, defaulting to 'channels_last' (NHWC)
        # The BatchNormalization axis is changed from 1 to 3 to target the channel axis, which is now last.
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(x_image)))
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv1)))
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv2)))
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid')(h_conv3)))
        
        h_conv4_flat = Flatten()(h_conv4)
        # The fully connected layers' batch norm axis remains 1, as they operate on flattened 1D data.
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))
        
        # Output Heads
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))