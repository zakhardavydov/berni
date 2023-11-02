import matplotlib.pyplot as plt

from typing import Optional


class BaseStats:

    def __init__(self, num_rounds: int, x_label: Optional[str] = None, y_label: Optional[str] = None):
        self._num_rounds = num_rounds
        self.x_label = x_label
        self.y_label = y_label

        self.samples = []
        self.x = list(range(num_rounds))

    def plot(self, ax):
        ax.plot(self.x, self.samples)
        if self.x_label:
            ax.set(xlabel=self.x_label)
        if self.y_label:
            ax.set(ylabel=self.y_label)
        return ax
