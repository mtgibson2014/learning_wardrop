# This defines the Braess environment with it's associated network.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

import argparse
import json
import os

import collections

import os
import os.path

class BaseNetwork(object):
    """
    Base class for all traffic networks.
    """
    @property
    def routes(self):
        raise NotImplementedError

    def calculate_ttime(self, flows):
        raise NotImplementedError

class BraessNetwork(BaseNetwork):
    """Stores the cost for all links. Handles calculating the cost of a path given action
       of every car.
    """
    def __init__(self):
        self.__links = {
            "AB": lambda f: 1 + (f/100),
            "AC": lambda _: 2,
            "BD": lambda _: 2,
            "CD": lambda f: 1 + (f/100),
            "BC": lambda _: 0.25
        } # Dictionary of links and their congestion functions
        self.__paths = {
            "ABD": ("AB", "BD"),
            "ACD": ("AC", "CD"),
            "ABCD": ("AB", "BC", "CD")
        } # Dictionaries of paths to links
        self.total_flow = 100  # 100 cars in total on this network
        return 
    
    @property
    def routes(self):
        """Gives a list of all possible paths in the network to the environment. 
           The environment could then assign an action number to each path. 
        """
        return ("ABD", "ACD", "ABCD")
    
    def calculate_ttime(self, flows):
        """Given a dictionary of paths and flows, this function returns a dictionary of 
           paths and travel time (secs), a.k.a ttime.
           
           Arg:
               flows (dictionary): A dictionare where the key correspond to a path in the network of one o-d pair
                                   and the value corresponds to the flow on that path. Flow will be a float
                                   between 0 and 1 represent the percent of flow. 
           
           Returns: 
               travel_times (list): A list of travel times, order matters 
                                    --> according to the order in my list of paths.
        """
        congestion = {}
        for path in flows:
            links = self.__paths[path]
            for link in links:
                if link not in congestion:
                    congestion[link] = 0
                congestion[link] += flows[path] * self.total_flow
        
        t_time = {}
        for path in flows:
            total_time = 0
            # Calculate travel time of path by adding the congestion time of every 
            # link in that path
            links = self.__paths[path]
            for link in links:
                t_time_func = self.__links[link]
                total_time += t_time_func(congestion[link])
            t_time[path] = total_time
        return t_time

class WalidNetwork(BaseNetwork):
    pass


# class DiscreteBraessEnv(BraessEnv):

#     def __init__(self, params=None):
#         super().__init__()
#         self.n_choices = 11
#         self.observation_space = MultiDiscrete([self.n_choices, self.n_choices, self.n_choices])
#         self.action_space = MultiDiscrete([ self.n_choices, self.n_choices, self.n_choices])

#     def step(self, action):
#         multiplier = 1/(self.n_choices-1)
#         return super().step(multiplier*action)
