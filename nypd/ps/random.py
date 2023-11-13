import random as rand

from nypd.agent import AbsAgent

from .abstract import AbsPartnerSelection


class RandomPairSelection(AbsPartnerSelection):

    def select(prev: list[list[AbsAgent]] | None, round: int, num_agents: int) -> list[list[AbsAgent]]:
        if prev:
            return rand.shuffle(prev)
        return [(i, j) for i in range(num_agents) for j in range(i + 1, num_agents)]
