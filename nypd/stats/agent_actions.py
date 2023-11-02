from .base import BaseStats


class AgentsActions(BaseStats):
    x_label = "Number of rounds"
    y_label = "Action"

    def __init__(self, num_rounds: int):
        super().__init__(num_rounds, self.x_label, self.y_label)
    
    def add_sample(self, sample):
        pass
