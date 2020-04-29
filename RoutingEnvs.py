from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym import Env, spaces
import unittest
import numpy as np 
import random

import argparse
import json
import os

import numpy as np
import math
import matplotlib.pyplot as plt

import collections

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

import gym
from gym.spaces import Discrete, Box, Dict, MultiDiscrete
from gym.envs.registration import EnvSpec, register

import os
import os.path


class BaseIterEnv(gym.Env):
    """Base routing environment for an iterative game.
    See https://github.com/openai/gym/blob/master/gym/core.py for more details on how to structure an env.
    """

    def set_first_state(self):
        """
        Should create the first observation and set the first state as self.state
        """
        raise NotImplementedError

    def reset(self):
        """
        Simply returns self.state. This assumes that self.state is constantly updated with the latest 
        state in the experiment.
        """
        raise NotImplementedError

    def step(self, action):
        """
        Takes the flow distribution from the agent (represented as the action), and gives the 
        new observation, cost for the action, an indicator of whether an episode of the environment is finished,
        and an info dictionary. 

        Also, updates self.state with the new routing observations.
        """
        raise NotImplementedError

    def render(self):
        """
        Renders a visual representation of our environments.
        """
        raise NotImplementedError
  

class RoutingEnv(BaseIterEnv):
    """Traffic routing environment that only handles one population of flow. 
       See https://github.com/openai/gym/blob/master/gym/core.py for more details.
    """
    
    def __init__(self, params):
        """
        Need to set the observation space, action space, and reward range. Then we set up the first state.

        Args:
            params: A dictionary of parameters for the environment. Should include:
                - "network": Traffic network for the environment.
                - "routes": A list of routes associated with this population.
        """
        # Store the parameters
        self.network = params["network"]
        self.routes = params["routes"]
        if params["optimum"] == "nash":
            self.reward = self.nash_reward
        else:
            self.reward = self.social_reward
        
        # Observation space contains each route and travel times
        self.num_routes = len(self.routes)
        self.observation_space = spaces.Box(low=0,high=float('+inf'),shape=(self.num_routes,),dtype=np.float32)
        
        # Action space contains each route and flow distribution of population (decimal)
        episilon = 0.000001  # To avoid agent from choosing all zeros for the flow
        self.action_space = action_spaces = spaces.Box(low=episilon,high=1,shape=(self.num_routes,),dtype=np.float32)
        
        self.reward_range = (-float('inf'), 0)

        # For the infinite game, set the first state
        self.__record_data= False
        self.set_first_state()

    def set_first_state(self):
        """
        Sets self.state to be the first set of observations - which is the travel times on each path with 0 flow. 

        For the initial observation, it should have the format ---
                    state = [<traveltime_ABD>, <traveltime_ACD>, <traveltime_ABCD>] = [2, 2, 0.25]
        """
        flows = {route: 1/self.num_routes for route in self.routes}
        initial_dict = self.network.calculate_ttime(flows)
        
        # Turn initial observation to an array
        t_0 = []
        for route in self.routes:
            t_0.append(initial_dict[route])
        
        # State will be an array
        self.state = t_0

    def reset(self):
        """Simply return the current state in the env"""     
        return self.state
    
    def step(self, action):
        """Run one timestep of the environment's dynamics. 
        
        Env Step Procedure:
            (1) takes in routing distribution - comes from the action, 
            (2) calculate travel times for each path given flow on each path, 
            (3) return the reward which is the negative of the travel time.
            
        Args:
            action (list): The action from the agent. This should be a list where each element 
                           is the proportion of flow for each corresponding paths. 
        
        Returns:
            next_observation (array): the travel times determined for each path
            reward (float): -1*travel_time_of_agent
            done (boolean): True
            info (dict): other information needed - don't really need now though
        """   
        print("I'm stepping into the env...")
        if sum(action) != 0:
            action /= sum(action)

        # Step the environment and give the next observation, reward, and done indicators
        obs = self.calc_path_traveltimes(action)

        rew = self.reward(action)
        # print("Action is: %r" % action)

        if self.__record_data:
            self.__record(self.travel_times_dict, self.action_dict, rew)

        done = True  # When this is True, it reaches the social optimum pretty quick. 
                      # When it's False, it stabilizes around 3 flows before going to the social optimum. 
        info = {}

        # Set the state to be the new observation
        self.state = obs
                
        return obs, rew, done, info

    def set_data_collection(self):
        self.__data = {"path_times": {route: [] for route in self.routes},
                       "path_flow": {route: [] for route in self.routes},
                       "rewards": []}
        self.__record_data = True

    def __record(self, times, flow, reward):
        times_data = self.__data["path_times"]
        flow_data = self.__data["path_flow"]
        for route in self.routes:
            times_data[route].append(times[route])
            flow_data[route].append(flow[route])
        self.__data["rewards"].append(reward)

    def graph(self):
        times_data = self.__data["path_times"]
        plt.subplot(221)
        for route in self.routes:
            plt.plot(times_data[route], label=route)
        plt.xlabel('iteration')
        plt.ylabel('travel time')
        plt.title('Travel time on each path (PPO)')
        plt.grid()
        plt.legend()
        plt.show()

        flow_data = self.__data["path_flow"]
        plt.subplot(222)
        for route in self.routes:
            plt.plot(flow_data[route], label=route)
        plt.xlabel('iteration')
        plt.ylabel('flow_percent')
        plt.title('Flow on each path (PPO)')
        plt.grid()
        plt.legend()
        plt.show()

        # Add plot for rewards later.

    def calc_path_traveltimes(self, path_flow_dis):
        """Calculates the travel time experienced on each path for a given flow distribution.
        
        Args: 
            path_flow_dis: A list representing the distribution of flow on each path
        """
        self.action_dict = {}
        for i in range(len(path_flow_dis)):
            self.action_dict[self.routes[i]] = path_flow_dis[i]
        
        # Calculate the travel times and store flow distributions and travel times
        self.travel_times_dict = self.network.calculate_ttime(self.action_dict)
        
        #Transform dictionary into list
        travel_times = []
        for route in self.routes:
            travel_times.append(self.travel_times_dict[route])
        return travel_times

    def nash_reward(self, path_flow_dis):
        action_dict = {}
        for i in range(len(path_flow_dis)):
            action_dict[self.routes[i]] = path_flow_dis[i]
        
        ros_value = self.network.calculate_ros(action_dict)
        return -1*ros_value

    def social_reward(self, path_flow_dis):
        """

        Args: 
            path_flow_dis: list of numbers, where each number is the percent of 
                           flow on each path in the network.
        """
        # Calculate the reward for the population - mean (negative) travel time (Edit for MA w/ different routes)
        cost = self.calc_path_traveltimes(path_flow_dis)
        reward = np.dot(np.array(path_flow_dis), 
                        -1*np.array(cost))
        return reward

    def render(self):
        return
    

class MultiagentRoutingEnv(BaseIterEnv):
    """Same as the single env but made more general. 
    """
    
     def __init__(self, params):
        """
        Need to set the observation space, action space, and reward range. Then we set up the first state.

        Args:
            params: A dictionary of parameters for the environment. Should include:
                - "network": Traffic network for the environment.
                - "routes": A list of routes associated with this population.
        """
        # Store the parameters
        self.network = params["network"]
        self.routes = params["routes"]
        if params["optimum"] == "nash":
            self.reward = self.nash_reward
        else:
            self.reward = self.social_reward
        
        # Observation space contains each route and travel times
        self.num_routes = len(self.routes)
        self.observation_space = spaces.Box(low=0,high=float('+inf'),shape=(self.num_routes,),dtype=np.float32)
        
        # Action space contains each route and flow distribution of population (decimal)
        episilon = 0.000001  # To avoid agent from choosing all zeros for the flow
        self.action_space = action_spaces = spaces.Box(low=episilon,high=1,shape=(self.num_routes,),dtype=np.float32)
        
        self.reward_range = (-float('inf'), 0)

        # For the infinite game, set the first state
        self.__record_data= False
        self.set_first_state()

    def set_first_state(self):
        """
        Sets self.state to be the first set of observations - which is the travel times on each path with 0 flow. 

        For the initial observation, it should have the format ---
                    state = [<traveltime_ABD>, <traveltime_ACD>, <traveltime_ABCD>] = [2, 2, 0.25]
        """
        flows = {route: 1/self.num_routes for route in self.routes}
        initial_dict = self.network.calculate_ttime(flows)
        
        # Turn initial observation to an array
        t_0 = []
        for route in self.routes:
            t_0.append(initial_dict[route])
        
        # State will be an array
        self.state = t_0

    def reset(self):
        """Simply return the current state in the env"""     
        return self.state
    
    def step(self, action):
        """Run one timestep of the environment's dynamics. 
        
        Env Step Procedure:
            (1) takes in routing distribution - comes from the action, 
            (2) calculate travel times for each path given flow on each path, 
            (3) return the reward which is the negative of the travel time.
            
        Args:
            action (list): The action from the agent. This should be a list where each element 
                           is the proportion of flow for each corresponding paths. 
        
        Returns:
            next_observation (array): the travel times determined for each path
            reward (float): -1*travel_time_of_agent
            done (boolean): True
            info (dict): other information needed - don't really need now though
        """   
        print("I'm stepping into the env...")
        if sum(action) != 0:
            action /= sum(action)

        # Step the environment and give the next observation, reward, and done indicators
        obs = self.calc_path_traveltimes(action)

        rew = self.reward(action)
        # print("Action is: %r" % action)

        if self.__record_data:
            self.__record(self.travel_times_dict, self.action_dict, rew)

        done = True  # When this is True, it reaches the social optimum pretty quick. 
                      # When it's False, it stabilizes around 3 flows before going to the social optimum. 
        info = {}

        # Set the state to be the new observation
        self.state = obs
                
        return obs, rew, done, info

    def set_data_collection(self):
        self.__data = {"path_times": {route: [] for route in self.routes},
                       "path_flow": {route: [] for route in self.routes},
                       "rewards": []}
        self.__record_data = True

    def __record(self, times, flow, reward):
        times_data = self.__data["path_times"]
        flow_data = self.__data["path_flow"]
        for route in self.routes:
            times_data[route].append(times[route])
            flow_data[route].append(flow[route])
        self.__data["rewards"].append(reward)

    def graph(self):
        times_data = self.__data["path_times"]
        plt.subplot(221)
        for route in self.routes:
            plt.plot(times_data[route], label=route)
        plt.xlabel('iteration')
        plt.ylabel('travel time')
        plt.title('Travel time on each path (PPO)')
        plt.grid()
        plt.legend()
        plt.show()

        flow_data = self.__data["path_flow"]
        plt.subplot(222)
        for route in self.routes:
            plt.plot(flow_data[route], label=route)
        plt.xlabel('iteration')
        plt.ylabel('flow_percent')
        plt.title('Flow on each path (PPO)')
        plt.grid()
        plt.legend()
        plt.show()

        # Add plot for rewards later.

    def calc_path_traveltimes(self, path_flow_dis):
        """Calculates the travel time experienced on each path for a given flow distribution.
        
        Args: 
            path_flow_dis: A list representing the distribution of flow on each path
        """
        self.action_dict = {}
        for i in range(len(path_flow_dis)):
            self.action_dict[self.routes[i]] = path_flow_dis[i]
        
        # Calculate the travel times and store flow distributions and travel times
        self.travel_times_dict = self.network.calculate_ttime(self.action_dict)
        
        #Transform dictionary into list
        travel_times = []
        for route in self.routes:
            travel_times.append(self.travel_times_dict[route])
        return travel_times

    def nash_reward(self, path_flow_dis):
        action_dict = {}
        for i in range(len(path_flow_dis)):
            action_dict[self.routes[i]] = path_flow_dis[i]
        
        ros_value = self.network.calculate_ros(action_dict)
        return -1*ros_value

    def social_reward(self, path_flow_dis):
        """

        Args: 
            path_flow_dis: list of numbers, where each number is the percent of 
                           flow on each path in the network.
        """
        # Calculate the reward for the population - mean (negative) travel time (Edit for MA w/ different routes)
        cost = self.calc_path_traveltimes(path_flow_dis)
        reward = np.dot(np.array(path_flow_dis), 
                        -1*np.array(cost))
        return reward

    def render(self):
        return