from .matrix import MatrixGame
from .registry import game_registry


class PDGame(MatrixGame):

    name = "pd"

    payoff_matrix = [
        [(3, 3), (0, 5)],
        [(5, 0), (1, 1)]
    ]

    def __init__(self):
        super().__init__(payoff_matrix=self.payoff_matrix)


game_registry.add(PDGame)
