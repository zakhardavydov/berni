from typing import Type
import matplotlib.pyplot as plt
import pandas as pd
import json
import re
import seaborn as sns
import numpy as np

from nypd.stats.collector import StatsCollector
from nypd.structures.experiment_setup import ExperimentSetup


class StatPlottingHelper:
    def __init__(self):
        pass

    @staticmethod
    def _parse_json_key(exp_info, key):
        if key == "params":
            pattern_str= f'"{key}"\s*:\s*{{[^{{}}]*}}'
        else:
            pattern_str = f'"{key}"\s*:\s*[^{{}}]*,'
            
        pattern = pattern_str
        match = re.findall(pattern, exp_info)
        print(match)
        return match.__str__()

    @staticmethod
    def _extract_label(match, key):
        if key == "params":
            pattern = r'{[^{}]*}'
            match_res = re.findall(pattern, match)
            match_res = [match[1:-1] for match in match_res]
            print(match_res)
            res_str = ", ".join(match_res)
        elif key=="num_rounds" or key=="num_agents":
            key_value_pairs = re.findall(r'"([^"]+)":\s*([0-9]+)', match)
            res_str = ""
            if key_value_pairs:
                for i in key_value_pairs:
                    print(i)
                key, value = key_value_pairs[0]
                res_str = f"{key}: {value}"
                print(res_str)
        else:
            key_value_pairs = re.findall(r'"([^"]+)":\s*"([^"]+)"', match)
            res_str = ""
            if key_value_pairs:
                key, value = key_value_pairs[0]
                res_str = f"{key}: {value}"
                print(res_str)
                

        return res_str

    @staticmethod
    def seaborn_create_plot_single_change(df: Type[pd.DataFrame], key="", title=""):
        hcr_fig, hcr_ax = plt.subplots(1)
        hcr_fig.suptitle(title)
        avg_fig, avg_ax = plt.subplots(1)
        avg_fig.suptitle(title)
        coop_fig, coop_ax = plt.subplots(1)
        coop_fig.suptitle(title)

        colour_palette = sns.color_palette()
        sns.lineplot(data=df, x="round", y="hcr", hue=key, ax=hcr_ax, legend='full', palette=colour_palette)
        sns.lineplot(data=df, x="round", y="avg", hue=key, ax=avg_ax, legend='full', palette=colour_palette)
        sns.lineplot(data=df, x="round", y="coop", hue=key, ax=coop_ax, legend='full', palette=colour_palette)
        sns.move_legend(hcr_ax, "upper right")
        sns.move_legend(avg_ax, "upper right")
        sns.move_legend(coop_ax, "upper right")
        coop_ax.set_ylim(0,1)

        return hcr_fig, avg_fig, coop_fig

    @staticmethod
    def seaborn_create_plot_single_change_combined(df: Type[pd.DataFrame], key="", title=""):
        fig, (hcr_ax, avg_ax, coop_ax) = plt.subplots(3)
        fig.suptitle(title)

        colour_palette = sns.color_palette()
        sns.lineplot(data=df, x="round", y="hcr", hue=key, ax=hcr_ax, legend='full', palette=colour_palette)
        sns.lineplot(data=df, x="round", y="avg", hue=key, ax=avg_ax, legend='full', palette=colour_palette)
        sns.lineplot(data=df, x="round", y="coop", hue=key, ax=coop_ax, legend='full', palette=colour_palette)
        handles, labels = hcr_ax.get_legend_handles_labels()
        fig.legend(handles, labels, title=key, loc="upper right")
        hcr_ax.get_legend().remove()
        avg_ax.get_legend().remove()
        coop_ax.get_legend().remove()
        coop_ax.set_ylim(0,1)

        return fig

    # DEPRECATED!!!!
    def create_plot_single_change(runs_dict, key, title=""):
        hcr_fig, hcr_ax = plt.subplots(1)
        hcr_fig.suptitle(title)
        avg_fig, avg_ax = plt.subplots(1)
        avg_fig.suptitle(title)
        coop_fig, coop_ax = plt.subplots(1)
        coop_fig.suptitle(title)

        cur_run_info: StatsCollector
        cur_exp_info: str

        # NOTE: refactory!!!
        # labels = []
        # print(labels)
        for idx, (cur_run, (cur_exp_info, cur_run_info)) in enumerate(runs_dict.items()):
            cur_params = StatPlottingHelper._parse_json_key(cur_exp_info, key)
            cur_label = StatPlottingHelper._extract_label(cur_params, key)
            full_df = cur_run_info.get_df("coop")

            sns.lineplot(x = np.arange(0,len(full_df['hcr'])), y='hcr', data = full_df, ax = hcr_ax, label=cur_label)
            sns.move_legend(hcr_ax, "upper right")
            hcr_ax.set_ylabel("hcr")
            hcr_ax.set_xlabel("num of games")

            sns.lineplot(x = np.arange(0,len(full_df['avg'])), y='avg', data = full_df, ax = avg_ax, label=cur_label)
            sns.move_legend(avg_ax, "upper right")
            avg_ax.set_ylabel("avg")
            avg_ax.set_xlabel("num of games")
  
            sns.lineplot(x = np.arange(0,len(full_df['coop'])), y="coop", data = full_df, ax = coop_ax, label=cur_label)
            sns.move_legend(coop_ax, "upper right")
            coop_ax.set_ylabel("coop")
            coop_ax.set_xlabel("num of games")

        return hcr_fig, avg_fig, coop_fig
    
    def create_plot_single_change_combined(runs_dict, key, title=""):
        """This one uses matplotlib
        """
        fig, (hcr_ax, avg_ax, coop_ax) = plt.subplots(3)
        
        cur_run_info: StatsCollector
        cur_exp_info: str

        # NOTE: refactory!!!
        fig.suptitle(title)
        labels = []
        print(labels)
        for idx, (cur_run, (cur_exp_info, cur_run_info)) in enumerate(runs_dict.items()):
            cur_params = StatPlottingHelper._parse_json_key(cur_exp_info, key)
            labels.append(StatPlottingHelper._extract_label(cur_params, key))

            full_df = cur_run_info.get_df("coop")
            hcr_df = full_df["hcr"]
            avg_df = full_df["avg"]
            coop_df = full_df["coop"]

            hcr_ax.plot(hcr_df)
            hcr_ax.set_ylabel("hcr")
            hcr_ax.set_xlabel("num of games")

            avg_ax.plot(avg_df)
            avg_ax.set_ylabel("avg")
            avg_ax.set_xlabel("num of games")
  
            coop_ax.plot(coop_df)
            coop_ax.legend()
            coop_ax.set_ylabel("coop")
            coop_ax.set_xlabel("num of games")

        fig.legend(labels)
        fig.tight_layout()
        return fig       