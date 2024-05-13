from typing import Type, List, Dict, Tuple, Optional
import numpy as np

from nypd.agent import BaseAgent
from nypd.environment import AbsEnv

from nypd.strategy import AbsStrategy

from ..structures import AgentConfig, AgentConfigs
from ..agent import agent_registry
from ..strategy.registry import registry


from .abs import AbsSeed


class NaiveSeed(AbsSeed):

    @staticmethod
    def __pick_strategy(registry, agent_type: Type[BaseAgent], allowed_strategy: Optional[Dict[str, float]] = None):

        if allowed_strategy:
            picked = np.random.choice(
                list(allowed_strategy.keys()), 1, p=list(allowed_strategy.values())
            )
            picked = picked[0]
            return picked, registry.registry[agent_type.name][picked]
        # NOTE: bad design here, assuming only q_agent will be enter the else case
        list_to_choice = list(registry[agent_type.name].items())
        picked_idx = np.random.choice(len(list_to_choice))
        picked = list_to_choice[picked_idx]
        return picked

    @staticmethod
    def get_st_count():
        st_count = {}
        st_list = []
        reg = registry.registry
        # NOTE: so inefficient!!!
        for cur_type in list(reg.keys()):
            for cur_st in reg[cur_type]:
                st_list.append(cur_st)

        st_count = {key: 0 for key in st_list}
        return st_count
    
    @staticmethod
    def pick_strategy(registry, picked_config: AgentConfig, constructor: Type[BaseAgent]):
        if isinstance(picked_config.strategy, Dict):
            st_id, st = NaiveSeed.__pick_strategy(registry, constructor, picked_config.strategy)
        elif isinstance(picked_config.strategy, str):
            st_id = picked_config.strategy
            st = registry.registry[constructor][st_id]
        else:
            st_id, st = NaiveSeed.__pick_strategy(registry, constructor, None)

        if isinstance(st, AbsStrategy):
            strategy = st
        else:
            strategy = st()
        return strategy, st_id

    def seed(
            self,
            registry,
            env: AbsEnv,
            agents: AgentConfigs,
            count: int,
    ) -> Tuple[List[BaseAgent], Dict[str, float]]:
        """
        Refactor the seed
        """

        st_count = self.get_st_count()
        created_agents = []
        agents_configs = [c.config for c in agents.configs]
        agents_probs = [c.ratio for c in agents.configs]
        for i in range(count):
            picked_config: AgentConfig = np.random.choice(
                agents_configs, 1, p=agents_probs
            )[0]

            constructor = agent_registry.registry[picked_config.type]

            strategy, st_id = self.pick_strategy(registry, picked_config, constructor)

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
