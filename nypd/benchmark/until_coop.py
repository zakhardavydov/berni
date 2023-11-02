from typing import Optional
from collections import deque

from ..stats import StatsCollector, CoopRate

from .abstract import AbsBenchmark


class UntilCoopBenchmark(AbsBenchmark):
    """
    Benchmark measures the number of rounds required for the game
    to converge to particular threshold of cooperation
    """

    def __init__(self, round_window_size: int, coop_threshold: float, interval_size: float):
        self.round_window_size = round_window_size
        self.coop_threshold = coop_threshold

        self.lower_coop = (1 - interval_size) * coop_threshold
        self.upper_coop = (1 + interval_size) * coop_threshold

    def apply(self, stats_collector: StatsCollector) -> Optional[float]:
        coop_stats: CoopRate = stats_collector.metrics.get("coop")
        if not coop_stats:
            raise ValueError("Coop metric has not been collected for this run")

        window_queue = deque(maxlen=self.round_window_size)

        for x, sample in zip(coop_stats.x, coop_stats.samples):
            window_queue.append(sample)

            av = (sum(window_queue) / self.round_window_size)

            if self.lower_coop < av < self.upper_coop:
                return x

        return None
