import pickle, argparse, json, math
import time
import json
import random

from utils import DEBUG_PRINT
from user_simulator import UserSimulator
from dqn_agent import DQNAgent
from state_tracker import StateTracker

if __name__ == "__main__":
    # Can provide constants file path in args OR run it as is and change 'CONSTANTS_FILE_PATH' below
    # 1) In terminal: python train.py --constants_path "constants.json"
    # 2) Run this file as is
    parser = argparse.ArgumentParser()
    parser.add_argument('--constants_path', dest='constants_path', type=str, default='')
    args = parser.parse_args()
    params = vars(args)

    # Load constants json into dict
    CONSTANTS_FILE_PATH = 'constants.json'
    if len(params['constants_path']) > 0:
        constants_file = params['constants_path']
    else:
        constants_file = CONSTANTS_FILE_PATH

    with open(constants_file) as f:
        constants = json.load(f)

    # Load file path constants
    file_path_dict = constants['db_file_paths']
    DATABASE_FILE_PATH = file_path_dict['database']
    DICT_FILE_PATH = file_path_dict['dict']
    USER_GOALS_FILE_PATH = file_path_dict['user_goals']
    DIALOG_FILE_PATH = file_path_dict['dialogs']

    # Load run constants
    run_dict = constants['run']
    USE_USERSIM = run_dict['usersim']
    NUM_EP_WARMUP = run_dict['num_ep_warmup'] # number of episode in warmup

    # Load product DB
    database= json.load(open(DATABASE_FILE_PATH, encoding='utf-8'))

    # Load movie dict
    db_dict = json.load(open(DICT_FILE_PATH, encoding='utf-8'))

    # Load goal File
    user_goals = json.load(open(USER_GOALS_FILE_PATH, encoding='utf-8'))

    # Load dialogs File
    dialogs = json.load(open(DIALOG_FILE_PATH, encoding='utf-8'))

    # Init. Objects
    if USE_USERSIM:
        user = UserSimulator(user_goals, constants, database)

    state_tracker = StateTracker(database, constants)
    dqn_agent = DQNAgent(constants)


def episode_reset():
    """
    Resets the episode/conversation in the warmup.

    Called in warmup to reset the state tracker, user and agent.

    """

    # First reset the state tracker
    state_tracker.reset()
    # Reset user
    user.reset()
    # Finally, reset agent
    dqn_agent.reset()


def warmup_run():
    """
    Runs the warmup stage of training which is used to fill the agents memory.

    Using dialogs created based on defined rules in order to induce the agent's initial behavior.
    Loop terminates when number of round is equal to NUM_EP_WARMUP or when the memory buffer is full.

    """

    print('Warmup Started...')
    total_step = 0
    for dialog in dialogs:
        if total_step == NUM_EP_WARMUP and dqn_agent.is_memory_full():
            break

        DEBUG_PRINT("Start conversation!!!")
        # Reset episode
        episode_reset()
        done = False
        # Run dialog
        num_round = 0
        for action in dialog:
            if action['speaker'] == 'user':
                user_action, reward, done, success = user.get_action(action, num_round)
                print("user:\t", user_action)
                DEBUG_PRINT("reward = %d" % reward + ", success = %s" % success)
            elif action['speaker'] == 'agent':
                agent_action, done = dqn_agent.get_action(action, num_round)
                print("agent:\t", agent_action)
            # Finish conversation
            if done:
                break

        total_step += 1

warmup_run()