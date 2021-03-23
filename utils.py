from inspect import getframeinfo, stack
import os

from dialogue_config import FAIL, SUCCESS

def DEBUG_PRINT(*arg):
    caller = getframeinfo(stack()[1][0])
    filename = os.path.basename(caller.filename)
    print("[%s][%s]" % (filename, caller.function), end =" ")
    for message in arg:
        print("%s" % (message), end ="") # python3 syntax print
    print("")

def reward_function(success, max_round):
    """
    Return the reward given the success.

    Return -1 + -max_round if success is FAIL, -1 + 2 * max_round if success is SUCCESS and -1 otherwise.

    Parameters:
        success (int)

    Returns:
        int: Reward
    """

    reward = -1
    if success == FAIL:
        reward += -max_round
    elif success == SUCCESS:
        reward += 2 * max_round
    return reward
