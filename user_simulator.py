import random, copy

from utils import DEBUG_PRINT
from utils import reward_function
from dialogue_config import FAIL, NO_OUTCOME, SUCCESS, UNSUITABLE

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

    def reset(self):
        """
        Resets the user sim. in warmup by ...

        """
        # False for failure, true for success, init. to failure
        self.constraint_check = SUCCESS

    def get_action(self, action, round_num):
        """
        Return the response of the user sim. to the agent by using action in defined dialog.
        Check if the agent has succeeded or lost or still going.

        Using in warmup.

        Parameters:
            action (dict): The user action that is picked in defined dialog.

        Returns:
            dict: User sim. response
            int: Reward
            bool: Done flag
            int: Success: -1, 0 or 1 for loss, neither win nor loss, win
        """

        done = False
        success = NO_OUTCOME

        # First check round num, if equal to max then fail
        if round_num == self.max_round:
            DEBUG_PRINT("max round reached")
            done = True
            success = FAIL
            user_response = {}
            user_response['intent'] = 'done'
            user_response['inform_slots'] = {}
            user_response['request_slots'] = {}
        else:
            user_response = {}
            user_response['intent'] = action['intent']
            user_response['inform_slots'] = copy.deepcopy(action['inform_slots'])
            user_response['request_slots'] = {}
            for slot in action['request_slots']:
                user_response['request_slots'][slot] = 'UNK'

            if user_response['intent'] == 'ok':
                self.constraint_check = SUCCESS
            elif user_response['intent'] == 'reject':
                self.constraint_check = FAIL
            elif user_response['intent'] == 'done':
                if self.constraint_check == SUCCESS:
                    done = True
                    success = SUCCESS
                else:
                    done = True
                    success = FAIL

        reward = reward_function(success, self.max_round)

        return user_response, reward, done, True if success == SUCCESS else False

