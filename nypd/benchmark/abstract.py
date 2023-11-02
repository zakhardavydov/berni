from abc import ABC, abstractmethod

from ..stats import StatsCollector


class AbsBenchmark(ABC):

    @abstractmethod
    def apply(self, stats_collector: StatsCollector) -> float:
        pass
