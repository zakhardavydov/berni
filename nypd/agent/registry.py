from typing import Dict, Type

from .base import BaseAgent


class AgentRegistry:

    def __init__(self):
        self.registry: Dict[str, Type[BaseAgent]] = {}

    def add(self, agent: Type[BaseAgent]):
        self.registry[agent.name] = agent


agent_registry = AgentRegistry()
