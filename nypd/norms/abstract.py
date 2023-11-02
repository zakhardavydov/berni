from abc import ABC, abstractmethod


class AbsNorm(ABC):

    name: str

    @abstractmethod
    def calculate_reward(self, agent, opponent):
        pass

    @property
    @abstractmethod
    def reward(self) -> float:
        pass
    
