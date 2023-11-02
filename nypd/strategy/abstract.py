from abc import ABC, abstractmethod

from ..agent import AbsAgent
from ..structures.action import Action


class AbsStrategy(ABC):

    id: str

    @abstractmethod
    def play(self, agent: AbsAgent) -> Action:
        pass
