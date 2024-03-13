import networkx as nx

from nypd.agent import AbsAgent, BaseAgent
from nypd.env.abstract import AbsEnv
from nypd.structures.agent_config import AgentConfigs

from .base_graph import BaseGraphPartnerSelection


class ErdosRenyiPartnerSelection(BaseGraphPartnerSelection):

    def __init__(self, grid_size: int, p: float, seed: int) -> None:
        super().__init__(grid_size)

        self._p = p
        self._seed_value = seed

    def seed(self, env: AbsEnv, agents: AgentConfigs) -> tuple[list[BaseAgent], dict[str, float]]:
        self._G = nx.erdos_renyi_graph(n=self._node_count, p=self._p, seed=self._seed_value)
        return super().seed(env, agents)
