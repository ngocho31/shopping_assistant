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

    def __init__(self, constants):
        """
        The constructor of DQNAgent.

        The constructor of DQNAgent which saves constants, sets up neural network graphs, etc.

        Parameters:
            state_size (int): The state representation size or length of numpy array
            constants (dict): Loaded constants in dict

        """

        self.max_memory_size = constants['agent']['max_mem_size']

        self.max_round = constants['run']['max_round_num'] # number of round (one time sentence user-agent) in episode

        # the agents memory
        self.memory = []

    def is_memory_full(self):
        """Returns true if the memory is full."""

        return len(self.memory) == self.max_memory_size

    def reset(self):
        """Resets ..."""

    def get_action(self, action, round_num):
        """
        Return the action of the agent by using action in defined dialog.
        Check if the agent has succeeded or lost or still going.

        Using in warmup.

        Parameters:
            action (dict): The user action that is picked in defined dialog.
            round_num (dict): The number of rounds have been taken.

        Returns:
            dict: Agent response
            bool: Done flag
        """

        done = False

        # First check round num, if equal to max then fail
        if round_num == self.max_round:
            DEBUG_PRINT("max round reached")
            done = True
            agent_response = {}
            agent_response['intent'] = 'done'
            agent_response['inform_slots'] = {}
            agent_response['request_slots'] = {}
        else:
            agent_response = {}
            agent_response['intent'] = action['intent']
            agent_response['inform_slots'] = copy.deepcopy(action['inform_slots'])
            agent_response['request_slots'] = {}
            for slot in action['request_slots']:
                agent_response['request_slots'][slot] = 'UNK'

            if agent_response['intent'] == 'done':
                done = True

        return agent_response, done

