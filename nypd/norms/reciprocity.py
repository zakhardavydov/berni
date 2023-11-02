from .base import BaseNorm
from .registry import norm_registry


class Reciprocity(BaseNorm):
    """
    Norm of reciprocity
    Cooperate on the first move and then imitate the last move of the opponent
    on every subsequent action
    """

    name = "reciprocity"

    def __init__(self, reward=2) -> None:
        super().__init__(reward=reward)

    def calculate_reward(self, agent, opponent):
        if agent.env.rounds < 2:
            # beggining of the game
            return 0
        if agent.action == opponent.last_action:
            return (self.reward * agent.norm_weight)
        return (-self.reward * agent.norm_weight)


norm_registry.add(Reciprocity)
