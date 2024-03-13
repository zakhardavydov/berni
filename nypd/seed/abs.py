from abc import ABC

from nypd.agent import BaseAgent
from nypd.env import AbsEnv
from nypd.structures import AgentConfigs


class AbsSeed(ABC):
    
    def seed(
            self,
            env: AbsEnv,
            agents: AgentConfigs
    ) -> tuple[list[BaseAgent], dict[str, float]]:
        raise NotImplementedError
