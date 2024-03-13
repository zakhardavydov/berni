from abc import ABC, abstractmethod
from typing import Optional, Dict, List

from ..structures.action import Action
from ..structures.agent_history_item import AgentHistoryItem


class AbsAgent(ABC):

    @abstractmethod
    def clean(self):
        pass

    @property
    @abstractmethod
    def opponent(self) -> Optional['AbsAgent']:
        pass

    @property
    @abstractmethod
    def memory(self) -> Dict['AbsAgent', List[Action]]:
        pass

    @property
    @abstractmethod
    def action(self) -> Optional[Action]:
        pass

    @property
    @abstractmethod
    def last_action(self) -> Optional[Action]:
        pass

    @property
    @abstractmethod
    def last_opponent(self) -> Optional['AbsAgent']:
        pass

    @property
    @abstractmethod
    def last_reward(self) -> Optional[float]:
        pass

    @property
    @abstractmethod
    def last_round(self) -> Optional[float]:
        pass

    @abstractmethod
    def get_last_history(self, buffer_size: int = 1) -> List[AgentHistoryItem]:
        pass

    @property
    @abstractmethod
    def env(self):
        pass

    @property
    @abstractmethod
    def norm_weight(self):
        pass

    @abstractmethod
    def act(self, opponent: Optional[int] = None):
        pass

    @abstractmethod
    def save_history(self, opponent, opponent_model, action, reward, round, bias_gap):
        pass

    @abstractmethod
    def observe(self, reward, state):
        pass
