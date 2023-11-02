from dataclasses import dataclass

from ..stats.collector import StatsCollector

from .experiment_setup import ExperimentSetup


@dataclass
class RunInfo:
    exp_info: ExperimentSetup
    collector: StatsCollector
