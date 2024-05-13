import random

from nypd.agent import AbsAgent
from .base_grid import BaseGridPartnerSelection


class RandomGridPartnerSelection(BaseGridPartnerSelection):

    def __init__(self, grid_size: int, seed: int) -> None:
        super().__init__(grid_size=grid_size)

    def _select(self, agent_index: int) -> int:
        neighbours = self.neighbours(agent_index=agent_index)
        return random.choice(neighbours)

    def select(self, prev: list[list[AbsAgent]] | None, round: int, num_agents: int) -> list[list[AbsAgent]]:
        matched = {}
        for agent_index in range(0, num_agents):
            matched[agent_index] = self._select(agent_index)
        out = [[src, target] for src, target in matched.items()]
        return out
