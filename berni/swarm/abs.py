import os

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

from nypd.structures import AgentConfigs
from nypd.agent import BaseAgent
from nypd.strategy import AbsStrategy, StrategyRegistry


class AbsSwarmGenerator(ABC):

    @staticmethod
    def get_prompts(path: str) -> dict[str, str]:
        paths = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(path)
            for f in filenames if os.path.splitext(f)[1] == ".txt"
        ]
        return {
            ".".join(str(Path(p).relative_to(path)).split(".")[:-1]): AbsSwarmGenerator.get_prompt(p) for p in paths
        }

    @staticmethod
    def get_prompt(path: str) -> str:
        return open(path, "r").read()

    @abstractmethod
    def generate(self, *args, **kwargs) -> AgentConfigs:
        raise NotImplementedError

    @staticmethod
    def register_strategy(
            registry: StrategyRegistry,
            strategies: list[tuple[Type[BaseAgent], Type[AbsStrategy] | AbsStrategy]]
    ):
        for agent, strategy in strategies:
            registry.add(agent, strategy)
