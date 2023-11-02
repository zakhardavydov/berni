from abc import ABC, abstractmethod
from typing import List
from typing import Optional

import numpy as np

from ..structures.action import Action


class AbsEnv(ABC):

    @abstractmethod
    def add_agent(self, agent):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @property
    @abstractmethod
    def gini_coefficient(self):
        pass

    @property
    @abstractmethod
    def prev_round_scores(self):
        pass

    @property
    @abstractmethod
    def current_payoffs(self):
        pass

    @property
    @abstractmethod
    def num_rounds(self) -> int:
        pass

    @property
    @abstractmethod
    def scores(self) -> np.array:
        pass

    @property
    @abstractmethod
    def rewards(self) -> np.array:
        pass

    @property
    @abstractmethod
    def actions(self) -> List[Action]:
        pass
