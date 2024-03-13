import random
import networkx as nx

from nypd.agent import AbsAgent, BaseAgent
from abc import ABC, abstractmethod


from nypd.agent import AbsAgent
from nypd.env.abstract import AbsEnv
from nypd.seed import AbsSeed
from nypd.structures.agent_config import AgentConfigs

from .structured import StructuredPartnerSelection


class BaseGraphPartnerSelection(AbsSeed, StructuredPartnerSelection, ABC):

    def __init__(self, grid_size: int) -> None:
        self._grid_size = grid_size
        self._node_count = grid_size * grid_size

        self._G: nx.Graph | None = None
    
    def neighbours(self, agent_index: int, depth: int = 1) -> list[int]:
        neighbors = set(self._G.neighbors(agent_index))
        if depth == 1:
            return list(neighbors)
        
        for current_depth in range(1, depth):
            for neighbor in list(neighbors):
                neighbors.update(self._G.neighbors(neighbor))
        
        neighbors.discard(agent_index)
        return list(neighbors)

    def seed(self, env: AbsEnv, agents: AgentConfigs, count: int) -> tuple[list[BaseAgent], dict[str, float]]:
        raise NotImplementedError

    def select(self, prev: list[list[AbsAgent]] | None, round: int, num_agents: int) -> list[tuple[AbsAgent, list[AbsAgent]]]:
        matched = {}
        for agent_index in range(0, num_agents):
            matched[agent_index] = self.neighbours(agent_index)
        out = [[src, target] for src, target in matched.items()]
        return out

    def num_agents(self) -> int:
        self._G.nodes
