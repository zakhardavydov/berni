from collections import defaultdict

from .base import BaseNorm
from .registry import norm_registry


class Reputation0(BaseNorm):
    """
    Norm 0 from Cooperation and Reputation Dynamics with Reinforcement Learning
    aka reputation has no impact on rewards
    """

    name = "reputation0"

    def __init__(self) -> None:
        super().__init__()

    def calculate_reward(self, agent, opponent):
        return 1


norm_registry.add(Reputation0)
