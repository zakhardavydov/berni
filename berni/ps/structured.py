from abc import ABC, abstractmethod

from nypd.ps import AbsPartnerSelection


class StructuredPartnerSelection(AbsPartnerSelection):

    @abstractmethod
    def num_agents(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def neighbours(self, depth: int = 1) -> list[int]:
        raise NotImplementedError
