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

    def __pick_strategy(self, agent_type: Type[BaseAgent], allowed_strategy: Optional[Dict[str, float]] = None):

        if allowed_strategy:
            picked = np.random.choice(
                list(allowed_strategy.keys()), 1, p=list(allowed_strategy.values())
            )
            picked = picked[0]
            return picked, self.registry[agent_type.name][picked]
        # NOTE: bad design here, assuming only q_agent will be enter the else case
        list_to_choice = list(self.registry[agent_type.name].items())
        picked_idx = np.random.choice(len(list_to_choice))
        picked = list_to_choice[picked_idx]
        return picked

    def _get_st_count(self):
        st_count = {}
        st_list = []
        # NOTE: so inefficient!!!
        for cur_type in list(self.registry.keys()):
            for cur_st in self.registry[cur_type]:
                st_list.append(cur_st)

        st_count = {key: 0 for key in st_list}
        return st_count

    def seed(
            self,
            env: AbsEnv,
            agents: AgentConfigs,
            count: int,
    ) -> Tuple[List[BaseAgent], Dict[str, float]]:
        """
        Refactor the seed
        """

        st_count = self._get_st_count()
        created_agents = []
        agents_configs = [c.config for c in agents.configs]
        agents_probs = [c.ratio for c in agents.configs]
        for i in range(count):
            picked_config: AgentConfig = np.random.choice(
                agents_configs, 1, p=agents_probs
            )[0]

            constructor = agent_registry.registry[picked_config.type]

            if isinstance(picked_config.strategy, Dict):
                st_id, st = self.__pick_strategy(constructor, picked_config.strategy)
            elif isinstance(picked_config.strategy, str):
                st_id = picked_config.strategy
                st = self.registry[constructor][st_id]
            else:
                st_id, st = self.__pick_strategy(constructor, None)

            if isinstance(st, AbsStrategy):
                strategy = st
            else:
                strategy = st()
            created = constructor(
                env=env,
                id=i,
                strategy=strategy,
                **picked_config.params
            )
            created_agents.append(created)
            st_count[st_id] += 1
        st_ratio = {key: value / count for key, value in st_count.items()}
        return created_agents, st_ratio


registry = StrategyRegistry()
