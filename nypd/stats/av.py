from typing import Optional

import mlflow
import numpy as np

from .base import BaseStats


class AvScore(BaseStats):

    name = "av-score"

    x_label = "Number of rounds"
    y_label = "Average Score"

    def __init__(self, num_rounds: int):
        super().__init__(num_rounds, self.x_label, self.y_label)

    def add_sample(self, sample, ml_flow_client: Optional[mlflow.MlflowClient] = None, run_id: str = None, tracked: bool = False):
        avg_score = np.sum(sample)
        self.samples.append(avg_score)

        if tracked and ml_flow_client and run_id:
            ml_flow_client.log_metric(run_id, self.name, avg_score)
