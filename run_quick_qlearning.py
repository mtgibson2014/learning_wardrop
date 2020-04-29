# Quick q-learning implementation 

import numpy as np
import gym
from gym.envs.registration import EnvSpec, register, registry

import random
from IPython.display import clear_output

register(id = "DiscreteBraessEnv-v0", entry_point = "learning_wardrop:DiscreteBraessEnv")
env = gym.make('DiscreteBraessEnv-v0')


q_table = {}


"""Training the agent"""

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()
    state_rep = str(state)
    if state_rep not in q_table:
        q_table[state_rep] = {}

    epochs, penalties, reward, = 0, 0, 0
    
    if random.uniform(0, 1) < epsilon:
        flow1 = env.action_space.sample() # Explore action space
        flow2 = env.action_space.sample()
        flow3 = env.action_space.sample()
        action = np.array([flow1, flow2, flow3])
        action_rep = str(action)
    else:
        # action = np.argmax(q_table[state_rep]) # Exploit learned values

    # Enter the action in the dictionary
    if action not in q_table[state_rep]:
        action_rep = str(action)
        q_table[state_rep][action_rep] = 0

    next_state, reward, done, info = env.step(action) 
    next_state_rep = str(next_state)
    if next_state_rep not in q_table:
        q_table[next_state_rep] = {}
    
    old_value = q_table[state_rep][action_rep]
    next_max = get_max_reward(q_table[next_state_rep])
    
    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    q_table[state_rep][action] = new_value

    if reward == -10:
        penalties += 1

    state = next_state
    epochs += 1
    print("Observation: %r\nAction: %r\nReward: %r\nNext State: %r" % (state, action, reward, next_state))
    # Then print the Q table next.

print("Training finished.\n")

def get_max_reward(action_dicts):
    pass