from typing import Optional

import mlflow


class RatioPie:
    def __init__(self, ratio: dict, ml_flow_client: Optional[mlflow.MlflowClient] = None, run_id: str = None):
        self.labels = list(ratio.keys())
        self.percent = list(ratio.values())

        if ml_flow_client and run_id:
            for label, percent in list(ratio.items()):
                ml_flow_client.log_param(run_id, f"strategy-ratio-{label}", percent)
    
    def plot(self, ax):
        ax.pie(self.percent, labels=self.labels, autopct='%.1f%%')
        return ax
