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

    def calculate_ros(self, flows):
        """ Calculates the rosenthal potential for this network using current routing distribution.

        Rosential potential function is: f(x) = sum_for_each_edge(integral of edge_cost(x), for x from 0 to flow on edge)
        
        The integral for the potential function was calculated on paper. It looks like:
                  f(x) = x_AB + (1/100)*(x_AB^2/2) + x_CD + (1/100)*(x_CD^2/2) + 2*x_AC + 2*x_BD + 0.25*x_BC

        Arg:
               flows (dictionary): A dictionare where the key correspond to a path in the network of one o-d pair
                                   and the value corresponds to the flow on that path. Flow will be a float
                                   between 0 and 1 represent the percent of flow.
        """
        congestion = {}
        for path in flows:
            links = self.__paths[path]
            for link in links:
                if link not in congestion:
                    congestion[link] = 0
                congestion[link] += flows[path] * self.total_flow

        x_AB = congestion["AB"]
        x_AC = congestion["AC"]
        x_BD = congestion["BD"]
        x_CD = congestion["CD"]
        x_BC = congestion["BC"]
        return x_AB + (1/100)*(x_AB**2/2) + x_CD + (1/100)*(x_CD**2/2) + 2*x_AC + 2*x_BD + 0.25*x_BC


class Network2(object):
    """Stores the cost for all links. Handles calculating the cost of a path given action
       of every car.
    """
    def __init__(self):
        self.__links = {
            "01": lambda f: f + 2., 
            "04": lambda f: f/2,
            "05": lambda f: f,
            "51": lambda f: f/3,
            "45": lambda f: 3*f, 
            "43": lambda f: f, 
            "24": lambda _: 0.5,
            "23": lambda f: f + 1.,
            "53": lambda f: f/4
        } # Dictionary of links and their congestion functions
        self.__paths = {
            "01": ("01","11"),
            "051": ("05", "51"),
            "0451": ("04", "45", "51"),
            "23": ("23","33"),
            "243": ("24","43"),
            "2453": ("24","45","53")
        } # Dictionaries of paths to links
        return 
    
    def paths(self, population):
        """Gives a list of all possible paths in the network to the environment. 
           The environment could then assign an action number to each path. 
        """
        if population == 0:
            return ("01", "051", "0451")
        elif population == 1:
            return ("23", "243", "2453")
        else:
            return "no such population"
        
    def shared_link(self): # a simple link for this example, need more generalized utility function for more comlicated networks
        return "45" 
    
    def calculate_ttime(self, flows): # flows now is a dictionary; add flow before feeding into the cost fct
        """Given a dictionary of paths and flows, this function returns a dictionary of 
           paths and travel time (secs), a.k.a ttime.
           
           Returns: 
               travel_times (dictionary): A dictionary of paths to their travel times
        """
        congestion = {}
        for population in flows:
            for path in flows[str(population)]:
                links = self.__paths[path]
                for link in links:
                    if link not in self.__links:
                        break
                    if link not in congestion:
                        congestion[link] = 0
                    congestion[link] += flows[str(population)][path]
                    
        t_time = {}
        for population in flows:
            t_time[population] = {}
            for path in flows[population]:
                total_time = 0
                # Calculate travel time of path by adding the congestion time of every 
                # link in that path
                links = self.__paths[path]
                for link in links:
                    if link not in self.__links:
                        break
                    t_time_func = self.__links[link]
                    total_time += t_time_func(congestion[link])
                t_time[population][path] = total_time
        
        return t_time
        
        
    def calculate_ttime_lambda(self, flows, Lambda):
        """Given a dictionary of paths and flows, this function returns a dictionary of 
           paths and travel time, considering the social factor lambda (secs).
           
           Returns: 
               travel_times (dictionary): A dictionary of paths to their travel times,
               considering the social factor lambda
        """
        congestion = {}
        for population in flows:
            for path in flows[str(population)]:
                links = self.__paths[path]
                for link in links:
                    if link not in self.__links:
                        break
                    if link not in congestion:
                        congestion[link] = 0
                    congestion[link] += flows[str(population)][path]
        
        t_time_lambda  = {}
        for population in flows:
            t_time_lambda[population] = {}
            for path in flows[population]:
                total_time = 0
                # Calculate travel time of path by adding the congestion time of every 
                # link in that path
                links = self.__paths[path]
                for link in links:
                    if link not in self.__links:
                        break
                    if link == "01" or link == "05" or link == "43" or link == "23":
                        total_time += Lambda * congestion[link]
                    elif link == "04":
                        total_time += Lambda * congestion[link] * 0.5
                    elif link == "51":
                        total_time += Lambda * congestion[link] / 3
                    elif link == "45":
                        total_time += Lambda * congestion[link] * 3
                    elif link == "53":
                        total_time += Lambda * congestion[link] / 4
                    t_time_func = self.__links[link]
                    total_time += t_time_func(congestion[link])
                t_time_lambda [population][path] = total_time
        return t_time_lambda

