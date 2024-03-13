from nypd.agent import AbsAgent

from .base_graph import BaseGraphPartnerSelection


class BarbasiAlbert(BaseGraphPartnerSelection):

    def __init__(self, grid_size: int) -> None:
        super().__init__(grid_size)

    def num_agents(self) -> int:
        return super().num_agents()
    
    def neighbours(self, agent_index: int, depth: int = 1) -> list[int]:
        return super().neighbours(agent_index, depth)
    
    def select(self, prev: list[list[AbsAgent]] | None, round: int, num_agents: int) -> list[list[AbsAgent]]:
        return super().select(prev, round, num_agents)
