from typing import List, Tuple

from ..structures.action import Action

from .abstract import AbsGame


class MatrixGame(AbsGame):

    name = "matrix"

    def __init__(self, payoff_matrix: List[List[Tuple[float, float]]]):
        self.payoff_matrix = payoff_matrix

    def get_payoff(self, action: Tuple[Action, Action]) -> Tuple[float, float]:
        return self.payoff_matrix[action[0]][action[1]]
