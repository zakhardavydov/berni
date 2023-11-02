from nypd.stats.collector import StatsCollector
import pickle as pkl
import matplotlib.pyplot as plt
import mpld3
import json
import pandas as pd
from nypd.structures import (
    ExperimentSetup
)

class ArtifactProcessor:
    def __init__(self, collector: StatsCollector, exp: ExperimentSetup):
        """ArtifactProcessor contains helper function to process data, helps data analysis

        Args:
            collector (StatsCollector): StatCollector that contains collected data
            exp (ExperimentSetup): experiment setup of corresponding collector 
        """
        self.collector = collector
        self.exp = exp
        self.plot_fig = None
        self._init_plot()

    def _init_plot(self):
        """Initialize plot for collector
        """
        metrics = self.collector.metrics
        keyword = ["hcr", "avg", "coop"]
        fig, _ = plt.subplots(len(keyword))
        fig.suptitle(f"INFO OF CURRENT RUN")

        all_axes = fig.get_axes()
        for r, cur_key in enumerate(keyword):
            cur_ax = all_axes[r]
            cur_tracker = metrics[cur_key]
            cur_ax.set_title(cur_tracker.name)
            cur_tracker.plot(cur_ax)
        fig.tight_layout()

        self.plot_fig = fig

    def to_csv(self, path: str):
        """Store current log info into csv.

        Args:
            path (str): path of the file
        """
        self.collector.get_df("coop").to_csv(path)

    def pkl_collector(self, path: str):
        """Pickle current collector

        Args:
            path (str): path of the file
        """
        with open(path, "wb") as f:
            pkl.dump(self.collector, f)

    def pkl_exp_info(self, path:str):
        """Pickle current expSetup

        Args:
            path (str): path of the file
        """
        with open(path, "wb") as f:
            pkl.dump(self.exp, f)    

    def save_exp_info_json(self, path: str):
        """Save current experiment info into json

        Args:
            path (str): path of the file
        """
        with open(path, "w") as f:   
            res = json.dumps(self.exp.to_json_dict())
            json.dump(res, f)

    def save_plot(self, path:str):
        """Save current plot into file

        Args:
            path (str): path of the file
        """
        if not self.plot_fig:
            print("ERROR: plot_fig -> None")
            return
        self.plot_fig.savefig(path)

    def save_html(self, path:str):
        """Save current plot into a html file

        Args:
            path (str): path of the file
        """
        if not self.plot_fig:
            print("ERROR: plot_fig -> None")
            return
        mpld3.save_html(self.plot_fig, path)



        

    