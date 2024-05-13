from abc import ABC

from nypd.strategy import StrategyRegistry
from nypd.agent import BaseAgent
from nypd.environment import AbsEnv
from nypd.structures import AgentConfigs


class AbsSeed(ABC):
    
    def seed(
            self,
            registry: StrategyRegistry,
            env: AbsEnv,
            agents: AgentConfigs
    ) -> tuple[list[BaseAgent], dict[str, float]]:
        raise NotImplementedError
