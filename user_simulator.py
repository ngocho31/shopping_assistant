import random, copy

from utils import DEBUG_PRINT

class UserSimulator:
    """Simulates a real user, to train the agent with reinforcement learning."""

    def __init__(self, goal_list, constants, database):
        """
        The constructor for UserSimulator. Sets dialogue config variables.

        Parameters:
            goal_list (list): User goals loaded from file
            constants (dict): Dict of constants loaded from file
            database (dict): The database in the format dict(long: dict)
        """

        self.goal_list = goal_list
        self.max_round = constants['run']['max_round_num'] # number of round (one time sentence user-agent) in episode

        # TEMP ----
        self.database = database
        # ---------

