from .base import BaseNorm
from collections import defaultdict

from .registry import norm_registry

"""
Norm 11 from Cooperation and Reputation Dynamics with Reinforcement Learning
aka Someone is “bad” only if they refuse to cooperate with a good individual.
0 is bad and 1 is good (inverse to action which is 0 for cooperate and 1 for defect)
"""


class Reputation3(BaseNorm):

    name = "reputation3"

    def __init__(self, reward=2) -> None:
        super().__init__(reward=reward)
        self.reputations = defaultdict(lambda: 0)

    def calculate_reward(self, agent, opponent):
        # if opponent is good and agent defects, give penalty assign bad.
        if (self.reputations[opponent.id] == 1 and agent.action == 1):
            self.reputations[agent.id] = 0
            return (-self.reward * agent.norm_weight)
        self.reputations[agent.id] = 1
        return (self.reward * agent.norm_weight)
        
norm_registry.add(Reputation3)