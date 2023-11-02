from dataclasses import dataclass

from .action import Action


@dataclass
class AgentHistoryItem:
    opponent: int
    action: Action
    reward: float
    round: int
