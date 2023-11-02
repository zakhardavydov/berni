from collections import defaultdict

from .base import BaseNorm
from .registry import norm_registry


class Reputation1(BaseNorm):
    """
    Norm 3 from Cooperation and Reputation Dynamics with Reinforcement Learning
    aka "good" if the agent cooperates and "bad" if they defect
    """

    name = "reputation1"

    def __init__(self, reward=2) -> None:
        super().__init__(reward=reward)
        self.reputations = defaultdict(lambda: 1)

    def calculate_reward(self, agent, opponent):
        self.reputations[agent.id] = agent.action
        if self.reputations[agent.id] == 0:
            return (self.reward * agent.norm_weight)
        else:
            return (-self.reward * agent.norm_weight)


norm_registry.add(Reputation1)
