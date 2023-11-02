from .base import BaseNorm
from .registry import norm_registry
from statistics import mean
import numpy as np


class InequityAversion(BaseNorm):
    """
    Inequity aversion using the Gini coefficient

    Calculates the Gini coefficient by calculating the impact of the agent's most recent action on the
    Gini coefficient of the previous round
    
    """
    name = "inequityAversion"

    def __init__(self, reward=2) -> None:
        super().__init__(reward=reward)
        self.pairs = []
        self.num_scores = 0
        self.rounds = 0

    def gini(self, x):
        x = np.asarray(x)
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    
    def calculate_reward(self, agent, opponent):
        if agent.env.rounds < 2:
            return 0
        scores = agent.env.prev_round_scores.copy()
        if self.rounds != agent.env.rounds:
            prev_g_coeff = self.gini(agent.env.prev_round_scores)
            agent.env.gini_coefficient = prev_g_coeff
            self.rounds = agent.env.rounds
        else:
            prev_g_coeff = agent.env.gini_coefficient
        payoff = agent.env.current_payoffs[agent.id]
        scores[agent.id] += payoff
        new_g_coeff = self.gini(scores)
        r = self.reward * agent.norm_weight
        if new_g_coeff > prev_g_coeff:
            r = -r
        if agent.env.rounds == 2:
            return 0
        return r
        

norm_registry.add(InequityAversion)
