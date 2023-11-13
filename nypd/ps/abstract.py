from abc import ABC

from nypd.agent import AbsAgent


class AbsPartnerSelection(ABC):
    
    def select(prev: list[list[AbsAgent]] | None, round: int, num_agents: int) -> list[list[AbsAgent]]:
        raise NotImplementedError
