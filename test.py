import pickle, argparse, json, math
import time
import json
import random

from utils import DEBUG_PRINT, SAVE_LOG
from user import User
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
        constants = json.load(f, encoding='utf-8')

    # Load file path constants
    file_path_dict = constants['db_file_paths']
    DATABASE_FILE_PATH = file_path_dict['database']
    SIZE_DATABASE_FILE_PATH = file_path_dict['size_database']
    DICT_FILE_PATH = file_path_dict['dict']
    SIZE_DICT_FILE_PATH = file_path_dict['size_dict']
    USER_GOALS_FILE_PATH = file_path_dict['user_goals']
    DIALOG_FILE_PATH = file_path_dict['dialogs']

    # Load run constants
    run_dict = constants['run']
    USE_USERSIM = run_dict['usersim']
    NUM_EP_WARMUP = run_dict['num_ep_warmup'] # number of episode in warmup
    NUM_EP_TRAIN = run_dict['num_ep_run']
    TRAIN_FREQ = run_dict['train_freq']
    SUCCESS_RATE_THRESHOLD = run_dict['success_rate_threshold']

    # Load product DB
    database= json.load(open(DATABASE_FILE_PATH, encoding='utf-8'))
    # Load size DB
    size_database= json.load(open(SIZE_DATABASE_FILE_PATH, encoding='utf-8'))

    # Load product dict
    # db_dict = json.load(open(DICT_FILE_PATH, encoding='utf-8'))

    # Load goal File
    user_goals = json.load(open(USER_GOALS_FILE_PATH, encoding='utf-8'))

    # Load dialogs File
    dialogs = json.load(open(DIALOG_FILE_PATH, encoding='utf-8'))

    # Init. Objects
    if USE_USERSIM:
        user = UserSimulator(user_goals, constants, database, size_database)
    else:
        user = User(constants)
    state_tracker = StateTracker(database, size_database, constants)
    dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)


def episode_reset():
    """
    Resets the episode/conversation in the training loops.

    Called in warmup and train to reset the state tracker, user and agent. Also get's the initial user action.

    """

    # First reset the state tracker
    state_tracker.reset()
    # Then pick an init user action
    user_action = user.reset()
    print("user: {}".format(str(user_action)))
    # And update state tracker
    state_tracker.update_state_user(user_action)
    # Finally, reset agent
    dqn_agent.reset()

def test_warmup():
    print('Testing Started...')
    episode_reset()
    ep_reward = 0
    done = False
    # Get initial state from state tracker
    state = state_tracker.get_state()
    while not done:
        # Agent takes action given state tracker's representation of dialogue
        agent_action_index, agent_action = dqn_agent.get_action_train(state)
        # Update state tracker with the agent's action
        state_tracker.update_state_agent_test(agent_action)
        print("agent: {}".format(str(agent_action)))

        # User takes action given agent action
        user_action, reward, done, success = user.step(agent_action)
        print("user: {}".format(str(user_action)))

        ep_reward += reward
        # Update state tracker with user action
        state_tracker.update_state_user(user_action)
        # Grab "next state" as state
        state = state_tracker.get_state(done)
    print('Episode: {} Success: {} Reward: {}'.format(0, success, ep_reward))


test_warmup()