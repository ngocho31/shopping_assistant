import pickle, argparse, json, math
import time
import json
import random

from utils import DEBUG_PRINT, SAVE_LOG
from user_simulator import UserSimulator
from error_model_controller import ErrorModelController
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
    USE_RULE = run_dict['use_rule'] # using based rule action

    # Load product DB
    database= json.load(open(DATABASE_FILE_PATH, encoding='utf-8'))

    # Load product dict
    db_dict = json.load(open(DICT_FILE_PATH, encoding='utf-8'))

    # Load goal File
    user_goals = json.load(open(USER_GOALS_FILE_PATH, encoding='utf-8'))

    # Load dialogs File
    dialogs = json.load(open(DIALOG_FILE_PATH, encoding='utf-8'))

    # Init. Objects
    if USE_USERSIM:
        user = UserSimulator(user_goals, constants, database)

    emc = ErrorModelController(db_dict, constants)
    state_tracker = StateTracker(database, constants)
    dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)


def episode_reset(user_action, use_rule=False):
    """
    Resets the episode/conversation in the warmup.

    Called in warmup to reset the state tracker, user and agent.

    """

    # First reset the state tracker
    state_tracker.reset()
    # Reset user
    # user.reset()
    if use_rule:
        user_action = user.reset_warmup(use_rule)
    else:
        user.reset_warmup()
        user.pick_action(user_action)
    DEBUG_PRINT(user_action)
    # Infuse with error
    emc.infuse_error(user_action)
    DEBUG_PRINT(user_action)
    # And update state tracker
    state_tracker.update_state_user(user_action)
    # Finally, reset agent
    dqn_agent.reset()


def run_round(state):
    # 1) Agent takes action given state tracker's representation of dialogue (state)
    agent_action_index, agent_action = dqn_agent.get_action_warmup(state)
    DEBUG_PRINT("agent:\t", agent_action)
    # 2) Update state tracker with the agent's action
    state_tracker.update_state_agent_warmup(agent_action, USE_RULE)
    SAVE_LOG("agent:\t", agent_action, filename='warmup.log')
    # 3) User takes action given agent action
    user_action, reward, done, success = user.step(agent_action)
    DEBUG_PRINT("user:\t", user_action)
    DEBUG_PRINT("reward:\t", reward)
    SAVE_LOG("user:\t", user_action, filename='warmup.log')
    SAVE_LOG("reward:\t", reward, filename='warmup.log')
    if not done:
        # 4) Infuse error into semantic frame level of user action
        emc.infuse_error(user_action)
        SAVE_LOG("user (error):\t", user_action, filename='warmup.log')
    # 5) Update state tracker with user action
    state_tracker.update_state_user(user_action)
    # 6) Get next state and add experience
    next_state = state_tracker.get_state(done)
    dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)

    return next_state, reward, done, success


def warmup_run():
    """
    Runs the warmup stage of training which is used to fill the agents memory.

    Using dialogs created based on defined rules in order to induce the agent's initial behavior.
    Loop terminates when number of round is equal to NUM_EP_WARMUP or when the memory buffer is full.

    """

    print('Warmup Started...')
    episode = 0
    while episode != NUM_EP_WARMUP and not dqn_agent.is_memory_full():
        SAVE_LOG("Start conversation!!!", filename='warmup.log')
        if USE_RULE:
            # Reset episode
            episode_reset(None, USE_RULE)
            done = False
            # Get initial state from state tracker
            state = state_tracker.get_state()
            while not done:
                next_state, _, done, _ = run_round(state)
                state = next_state
            episode += 1
        else:
            if episode == len(dialogs):
                episode = 0
            dialog = dialogs[episode]
            # Reset episode
            episode_reset(dialog[0])
            done = False
            reward_total = 0
            # Pick an first user action
            # user_action, reward, done, success = user.pick_action(dialog[0])
            # SAVE_LOG("user:\t", user_action, filename='warmup.log')
            # DEBUG_PRINT("user:\t", user_action)
            # DEBUG_PRINT("reward = %d" % reward + ", success = %s" % success)
            # Add user action into history
            # state_tracker.update_state_user(user_action)
            # # After user action, get state from state tracker
            # state = state_tracker.get_state(done)
            # for i, action in enumerate(dialog):
            #     if i == 0:
            #         continue
            #     if action['speaker'] == 'user':
            #         # Pick an user action in defined dialog
            #         user_action, reward, done, success = user.pick_action(action, agent_action)
            #         SAVE_LOG("user:\t", user_action, filename='warmup.log')
            #         SAVE_LOG("success: ", success, ", reward: ", reward, filename='warmup.log')
            #         reward_total += reward
            #         DEBUG_PRINT("user:\t", user_action)
            #         DEBUG_PRINT("reward = %d" % reward + ", success = %s" % success)
            #         # Add user action into history
            #         state_tracker.update_state_user(user_action)
            #         # After user action, get state from state tracker
            #         next_state = state_tracker.get_state(done)
            #         dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)
            #         state = next_state
            #         # Finish conversation
            #         if done:
            #             break
            #     elif action['speaker'] == 'agent':
            #         # Pick an agent action in defined dialog
            #         agent_action_index, agent_action = dqn_agent.pick_action(action)
            #         SAVE_LOG("agent:\t", agent_action, filename='warmup.log')
            #         DEBUG_PRINT("agent:\t", agent_action)
            #         # Add agent action into history
            #         state_tracker.update_state_agent(agent_action)
            # SAVE_LOG("success: ", success, ", reward total: ", reward_total, filename='warmup.log')

            episode += 1

    print('...Warmup Ended')
    # After fill the agents memory, train model based on them, to initial agent's behavior
    dqn_agent.train()
    dqn_agent.save_weights()


warmup_run()
