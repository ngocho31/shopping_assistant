from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random, copy
import numpy as np
import re

from utils import DEBUG_PRINT

# Some of the code based off of https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
# Note: In original paper's code the epsilon is not annealed and annealing is not implemented in this code either

class DQNAgent:
    """The DQN agent that interacts with the user."""

    def __init__(self, state_size, constants):
        """
        The constructor of DQNAgent.

        The constructor of DQNAgent which saves constants, sets up neural network graphs, etc.

        Parameters:
            state_size (int): The state representation size or length of numpy array
            constants (dict): Loaded constants in dict

        """

