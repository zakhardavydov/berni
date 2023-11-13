from typing import Type, List, Dict, Tuple, Optional
import numpy as np

from nypd.agent import BaseAgent
from nypd.env import AbsEnv

from .abstract import AbsStrategy

from ..structures import AgentConfig, AgentConfigs
from ..agent import agent_registry


class StrategyRegistry:

    def __init__(self):
        # NOTE: dqn could be merged into q_learning after finsh implementing dqn.
        # self.registry: Dict[str, Dict[str, Type[AbsStrategy]]] = {"static": {}, "q_learning": {}, "dqn": {}} 
        self.registry: Dict[str, Dict[str, Type[AbsStrategy]]] = {}
 
    def add(self, agent_type: Type[BaseAgent], strategy: Type[AbsStrategy]):
        if agent_type.name not in self.registry:
            self.registry[agent_type.name] = {}
        self.registry[agent_type.name][strategy.id] = strategy


registry = StrategyRegistry()
