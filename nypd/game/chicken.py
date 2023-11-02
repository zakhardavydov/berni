from .matrix import MatrixGame
from .registry import game_registry


class ChickensGame(MatrixGame):

    name = "chicken"

    payoff_matrix = [
        [(3, 3), (1, 4)],
        [(4, 1), (0, 0)]
    ]

    def __init__(self):
        super().__init__(payoff_matrix=self.payoff_matrix)


game_registry.add(ChickensGame)
