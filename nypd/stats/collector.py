

from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from nypd.stats.agent_ratio import RatioPie

from nypd.stats.base import BaseStats


class StatsCollector:

    def __init__(self, metrics: Dict[str, Any]):
        self.metrics = metrics
        self.stat_df = self._init_df(metrics)
        self.ratio_tracker = None

    def _init_df(
            self,
            metrics: Dict[str, Any],
            run_id: Optional[str] = None,
            params: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        df_dict = metrics.copy()
        for cur_metric, cur_tracker in list(metrics.items()):
            if "round" not in df_dict:
                df_dict["round"] = [i for i in range(0, len(cur_tracker.samples))]
            if run_id and "run_id" not in df_dict:
                df_dict["run_id"] = [run_id for _ in range(0, len(cur_tracker.samples))]
            if params:
                for i, config in enumerate(params):
                    if "name" in config["config"] and config["config"]["name"]:
                        prefix = config["config"]["name"]
                    else:
                        prefix = i
                    if "raio" in config:
                        df_dict[f"{prefix}__ratio"] = config["raio"]
                    if "ratio" in config:
                        df_dict[f"{prefix}__ratio"] = config["ratio"]
                    for param_name, param_value in config["config"]["params"].items():
                        col_name = f"{prefix}__{param_name}"
                        if col_name not in df_dict:
                            df_dict[col_name] = [param_value for _ in range(0, len(cur_tracker.samples))]
            # extract sample from tracker
            if isinstance(cur_tracker, RatioPie):
                for label, percent in zip(cur_tracker.labels, cur_tracker.percent):
                    df_dict[f"{cur_metric}__{label}"] = percent
                df_dict[cur_metric] = f"label -> {str(list(cur_tracker.labels))}, \
                                        percent -> {str(list(cur_tracker.percent))}"
            else:
                df_dict[cur_metric] = cur_tracker.samples
        return pd.DataFrame(df_dict)

    def add_metric(self, name: str, tracker: Any):
        self.metrics[name] = tracker
        # NOTE: not efficient, but should work
        self.stat_df = self._init_df(self.metrics)

    def get_df(self, y_col: str) -> pd.DataFrame:
        df = self.stat_df
        rounds = df["round"].tolist()
        y = df[y_col].tolist()
        lowess = sm.nonparametric.lowess(y, rounds, frac=0.05)
        df[y_col] = lowess[:, 1]
        return df

    def get_tracker(self, name:str) -> BaseStats:
        try:
            return self.metrics[name]
        except KeyError:
            pass

    def plot(self):
        fig, subplots = plt.subplots(4)
        for tracked, subplot in zip(list(self.metrics.values()), subplots):
            tracked.plot(subplot)
