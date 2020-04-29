# Includes the following algorithms:
#   Q-Learning algorithm
#   Sarsa

from stable_baselines.common.base_class import OffPolicyRLModel

class BaseAlgo():
    """
    Not sure if we need our own base class or if we should just inherit from stable_baseline's
    BaseRLModel in stable_baselines/common/base_class.py
    """
    pass


class QLearning(OffPolicyRLModel):
    pass