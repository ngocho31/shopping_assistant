import copy
import time
import numpy as np

from utils import DEBUG_PRINT

class StateTracker:
    """Tracks the state of the episode/conversation and prepares the state representation for the agent."""

    def __init__(self, database, constants):
        """
        The constructor of StateTracker.

        The constructor of StateTracker which creates a DB query object, creates necessary state rep. dicts, etc. and
        calls reset.

        Parameters:
            database (dict): The database with format dict(long: dict)
            constants (dict): Loaded constants in dict

        """

