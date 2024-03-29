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
    SIZE_DATABASE_FILE_PATH = file_path_dict['size_database']
    DICT_FILE_PATH = file_path_dict['dict']
    SIZE_DICT_FILE_PATH = file_path_dict['size_dict']
    USER_GOALS_FILE_PATH = file_path_dict['user_goals']
    DIALOG_FILE_PATH = file_path_dict['dialogs']

    # Load run constants
    run_dict = constants['run']
    USE_USERSIM = run_dict['usersim']
    NUM_EP_WARMUP = run_dict['num_ep_warmup'] # number of episode in warmup
    USE_RULE = run_dict['use_rule'] # using based rule action
    NUM_EP_TRAIN = run_dict['num_ep_run']
    TRAIN_FREQ = run_dict['train_freq']
    SUCCESS_RATE_THRESHOLD = run_dict['success_rate_threshold']

    # Load product DB
    database= json.load(open(DATABASE_FILE_PATH, encoding='utf-8'))
    # Load size DB
    size_database= json.load(open(SIZE_DATABASE_FILE_PATH, encoding='utf-8'))

    # Load product dict
    db_dict = json.load(open(DICT_FILE_PATH, encoding='utf-8'))
    # Load size dict
    size_db_dict = json.load(open(SIZE_DICT_FILE_PATH, encoding='utf-8'))

    # Load goal File
    user_goals = json.load(open(USER_GOALS_FILE_PATH, encoding='utf-8'))

    # Load dialogs File
    dialogs = json.load(open(DIALOG_FILE_PATH, encoding='utf-8'))

    # Init. Objects
    if USE_USERSIM:
        user = UserSimulator(user_goals, constants, database, size_database)

    emc = ErrorModelController(db_dict, size_db_dict, constants)
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
    user_action = user.reset_train()
    # Infuse with error
    emc.infuse_error(user_action)
    DEBUG_PRINT("user:\t", user_action)
    SAVE_LOG("user:\t", user_action, filename='test.log')
    # And update state tracker
    state_tracker.update_state_user(user_action)
    # Finally, reset agent
    dqn_agent.reset()


def episode_reset_warmup(user_action=None, use_rule=False):
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
        # DEBUG_PRINT(user_action)
        # Infuse with error
        emc.infuse_error(user_action)
    else:
        user.reset_warmup()
        user.pick_action(user_action)
    # DEBUG_PRINT(user_action)
    SAVE_LOG("user:\t", user_action, filename='warmup.log')
    DEBUG_PRINT("user:\t", user_action)
    # And update state tracker
    state_tracker.update_state_user(user_action)
    # Finally, reset agent
    dqn_agent.reset()


def run_round(state):
    # 1) Agent takes action given state tracker's representation of dialogue (state)
    agent_action_index, agent_action = dqn_agent.get_action_train(state)
    DEBUG_PRINT("agent:\t", agent_action)
    # 2) Update state tracker with the agent's action
    state_tracker.update_state_agent_train(agent_action)

    DEBUG_PRINT("agent:\t", agent_action)
    SAVE_LOG("agent:\t", agent_action, filename='test.log')

    # 3) User takes action given agent action
    user_action, reward, done, success = user.step(agent_action)
    DEBUG_PRINT("user:\t", user_action)
    if not done:
        # 4) Infuse error into semantic frame level of user action
        emc.infuse_error(user_action)
    DEBUG_PRINT("user (error):\t", user_action)
    SAVE_LOG("user (error):\t", user_action, filename='test.log')
    # 5) Update state tracker with user action
    state_tracker.update_state_user(user_action)
    # 6) Get next state and add experience
    next_state = state_tracker.get_state(done)
    dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)

    return next_state, reward, done, success


def run_round_warmup(state, agent_action=None, user_action=None, use_rule=False):
    # 1) Agent takes action given state tracker's representation of dialogue (state)
    if use_rule:
        agent_action_index, agent_action = dqn_agent.get_action_warmup(state)
    else:
        # Pick an agent action in defined dialog
        agent_action_index, agent_action = dqn_agent.pick_action(agent_action)
    DEBUG_PRINT("agent:\t", agent_action)
    # 2) Update state tracker with the agent's action
    state_tracker.update_state_agent_warmup(agent_action, use_rule)
    SAVE_LOG("agent:\t", agent_action, filename='warmup.log')
    # DEBUG_PRINT("agent:\t", agent_action)
    # 3) User takes action given agent action
    if use_rule:
        user_action, reward, done, success = user.step(agent_action)
    else:
        # Pick an user action in defined dialog
        user_action, reward, done, success = user.pick_action(user_action, agent_action)
    # DEBUG_PRINT("user:\t", user_action)
    # DEBUG_PRINT("reward:\t", reward)
    SAVE_LOG("reward:\t", reward, filename='warmup.log')
    if not done:
        # 4) Infuse error into semantic frame level of user action
        emc.infuse_error(user_action)
        # DEBUG_PRINT("user (error):\t", user_action)
        SAVE_LOG("user (error):\t", user_action, filename='warmup.log')
    DEBUG_PRINT("user:\t", user_action)
    # 5) Update state tracker with user action
    state_tracker.update_state_user(user_action)
    # 6) Get next state and add experience
    next_state = state_tracker.get_state(done)
    dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)

    return next_state, reward, done, success


def warmup_run():
    """
    Runs the warmup stage of training which is used to fill the agents memory.

    The agent uses it's rule-based policy to make actions. The agent's memory is filled as this runs.
    Loop terminates when the size of the memory is equal to WARMUP_MEM or when the memory buffer is full.

    """

    print('Warmup Started...')
    episode = 0
    while episode != NUM_EP_WARMUP and not dqn_agent.is_memory_full():
        SAVE_LOG("Start conversation!!!", filename='warmup.log')
        if USE_RULE:
            # Reset episode
            episode_reset_warmup(use_rule=USE_RULE)
            done = False
            # Get initial state from state tracker
            state = state_tracker.get_state()
            while not done:
                next_state, _, done, _ = run_round_warmup(state, use_rule=USE_RULE)
                state = next_state
            episode += 1
        else:
            if episode == len(dialogs):
                episode = 0
            dialog = dialogs[episode]
            # Reset episode
            episode_reset_warmup(dialog[0])
            done = False
            # Get initial state from state tracker
            state = state_tracker.get_state()
            i = 1
            while not done:
                next_state, _, done, _ = run_round_warmup(state=state, agent_action=dialog[i], user_action=dialog[i+1])
                state = next_state
                i += 2
            episode += 1

    print('...Warmup Ended')


def train_run():
    """
    Runs the loop that trains the agent.

    Trains the agent on the goal-oriented chatbot task. Training of the agent's neural network occurs every episode that
    TRAIN_FREQ is a multiple of. Terminates when the episode reaches NUM_EP_TRAIN.

    """

    print('Training Started...')
    episode = 0
    period_reward_total = 0
    period_success_total = 0
    success_rate_best = 0

    while episode < NUM_EP_TRAIN:
        SAVE_LOG("Start conversation!!!", filename='test.log')
        episode_reset()
        episode += 1
        done = False
        state = state_tracker.get_state()
        while not done:
            next_state, reward, done, success = run_round(state)
            period_reward_total += reward
            state = next_state
        DEBUG_PRINT("success: ", success)
        # SAVE_LOG("success: ", success, ", reward total: ", period_reward_total, filename='train.log')

        period_success_total += success

        # Train
        if episode % TRAIN_FREQ == 0:
            # Check success rate
            success_rate = period_success_total / TRAIN_FREQ
            avg_reward = period_reward_total / TRAIN_FREQ
            DEBUG_PRINT("episode: ", episode, ", success_rate = ", success_rate)
            SAVE_LOG("episode: ", episode, ", success rate: ", success_rate, filename='train.log')

            # Flush
            if success_rate >= success_rate_best and success_rate >= SUCCESS_RATE_THRESHOLD:
                DEBUG_PRINT("episode: ", episode, ", success_rate > threshold = ", success_rate)
                dqn_agent.empty_memory()
            # Update current best success rate
            if success_rate > success_rate_best:
                SAVE_LOG("Episode: ", episode, ", NEW BEST SUCCESS RATE: ", success_rate, ", Avg Reward: ", avg_reward, filename='train.log')
                DEBUG_PRINT("episode: ", episode, ", new best success_rate = ", success_rate)
                success_rate_best = success_rate
                dqn_agent.save_weights()
            period_success_total = 0
            period_reward_total = 0
            # Copy
            dqn_agent.copy()
            # Train
            dqn_agent.train()

    print('...Training Ended')

warmup_run()
train_run()