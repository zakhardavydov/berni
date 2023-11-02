from collections import defaultdict

from .base import BaseNorm
from .registry import norm_registry

class Reputation2(BaseNorm):
    """
    Norm 9 from Cooperation and Reputation Dynamics with Reinforcement Learning
    aka "good" if the agent cooperates with good agents and defects to bad agents
    0 is bad and 1 is good (inverse to action which is 0 for cooperate and 1 for defect)
    """

    name = "reputation2"

    def __init__(self, reward=2) -> None:
        super().__init__(reward=reward)
        self.reputations = defaultdict(lambda: 0)

    def calculate_reward(self, agent, opponent):
        # if cooperates and opponent is good or defect and opponent is bad, give bonus
        if (agent.action == 0 and self.reputations[opponent.id] == 1) or (agent.action == 1 and self.reputations[opponent.id] == 0):
            self.reputations[agent.id] = 1
            return (self.reward * agent.norm_weight)
        self.reputations[agent.id] = 0
        return (-self.reward * agent.norm_weight)


norm_registry.add(Reputation2)
