import networkx as nx

from .base_graph import BaseGraphPartnerSelection


class BarabasiAlbertPartnerSelection(BaseGraphPartnerSelection):

    def __init__(self, grid_size: int, m: int, seed: int) -> None:
        self._m = m
        self._seed_value = seed

        super().__init__(grid_size)

    def init_graph(self):
        self._G = nx.barabasi_albert_graph(n=self._node_count, m=self._m, seed=self._seed_value)
