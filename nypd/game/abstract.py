from abc import ABC, abstractmethod
from typing import Tuple

from ..structures.action import Action


class AbsGame(ABC):

    name: str

    @abstractmethod
    def get_payoff(self, action: Tuple[Action, Action]) -> Tuple[float, float]:
        pass
