import networkx as nx

from .base_graph import BaseGraphPartnerSelection


class WattsStrogatzPartnerSelection(BaseGraphPartnerSelection):

    def __init__(self, grid_size: int, k: int, beta: float, seed: int) -> None:
        self._k = k
        self._beta = beta
        self._seed_value = seed

        super().__init__(grid_size)

    def init_graph(self):
        self._G = nx.watts_strogatz_graph(n=self._node_count, k=self._k, p=self._beta, seed=self._seed_value)
