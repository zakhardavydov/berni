import networkx as nx

from .base_graph import BaseGraphPartnerSelection


class ErdosRenyiPartnerSelection(BaseGraphPartnerSelection):

    def __init__(self, grid_size: int, p: float, seed: int) -> None:
        self._p = p
        self._seed_value = seed

        super().__init__(grid_size)

    def init_graph(self):
        self._G = nx.erdos_renyi_graph(n=self._node_count, p=self._p, seed=self._seed_value)
