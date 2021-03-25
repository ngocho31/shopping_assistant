import pickle, argparse, json, math
import time
import json
import random

from utils import DEBUG_PRINT, SAVE_LOG
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
        constants = json.load(f, encoding='utf-8')

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
    NUM_EP_TRAIN = run_dict['num_ep_run']
    TRAIN_FREQ = run_dict['train_freq']
    SUCCESS_RATE_THRESHOLD = run_dict['success_rate_threshold']

    # Load product DB
    database= json.load(open(DATABASE_FILE_PATH, encoding='utf-8'))

    # Load product dict
    # db_dict = json.load(open(DICT_FILE_PATH, encoding='utf-8'))

    # Load goal File
    user_goals = json.load(open(USER_GOALS_FILE_PATH, encoding='utf-8'))

    # Load dialogs File
    dialogs = json.load(open(DIALOG_FILE_PATH, encoding='utf-8'))

    # Init. Objects
    if USE_USERSIM:
        user = UserSimulator(user_goals, constants, database)

    state_tracker = StateTracker(database, constants)
    dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)


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
        if total_step == NUM_EP_WARMUP or dqn_agent.is_memory_full():
            break

        SAVE_LOG("Start conversation!!!")
        # Reset episode
        episode_reset()
        done = False
        reward_total = 0
        # Pick an first user action
        user_action, reward, done, success = user.pick_action(dialog[0])
        SAVE_LOG("user:\t", user_action)
        # DEBUG_PRINT("reward = %d" % reward + ", success = %s" % success)
        # Add user action into history
        state_tracker.update_state_user(user_action)
        # After user action, get state from state tracker
        state = state_tracker.get_state(done)
        for i, action in enumerate(dialog):
            if i == 0:
                continue
            if action['speaker'] == 'user':
                # Pick an user action in defined dialog
                user_action, reward, done, success = user.pick_action(action, agent_action)
                SAVE_LOG("user:\t", user_action)
                SAVE_LOG("success: ", success, ", reward: ", reward)
                reward_total += reward
                # DEBUG_PRINT("reward = %d" % reward + ", success = %s" % success)
                # Add user action into history
                state_tracker.update_state_user(user_action)
                # After user action, get state from state tracker
                next_state = state_tracker.get_state(done)
                dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)
                state = next_state
                # Finish conversation
                if done:
                    break
            elif action['speaker'] == 'agent':
                # Pick an agent action in defined dialog
                agent_action_index, agent_action = dqn_agent.pick_action(action)
                SAVE_LOG("agent:\t", agent_action)
                # Add agent action into history
                state_tracker.update_state_agent(agent_action)
        SAVE_LOG("success: ", success, ", reward total: ", reward_total)

        total_step += 1

    # After fill the agents memory, train model based on them, to initial agent's behavior
    dqn_agent.train()
    dqn_agent.save_weights()


warmup_run()