from typing import Optional

import mlflow
import numpy as np

from .base import BaseStats


class CoopRate(BaseStats):

    name = "coop"

    x_label = "Number of rounds"
    y_label = "Cooperation rate"

    def __init__(self, num_rounds: int):
        super().__init__(num_rounds, self.x_label, self.y_label)

    def add_sample(self, sample, ml_flow_client: Optional[mlflow.MlflowClient] = None, run_id: str = None, tracked: bool = False):
        cooperation_rate = 1 - (np.sum(sample) / (len(sample)*2))
        self.samples.append(cooperation_rate)

        if tracked and ml_flow_client and run_id:
            ml_flow_client.log_metric(run_id, self.name, cooperation_rate)
