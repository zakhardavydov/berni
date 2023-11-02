import random
from typing import Tuple

from ..agent.abstract import AbsAgent
from ..structures.action import Action

from .abstract import AbsStrategy
from .registry import registry
from ..agent import StaticAgent


class Good(AbsStrategy):
    id = "good"

    def play(self, agent: AbsAgent) -> Action:
        return Action.C


class Bad(AbsStrategy):
    id = "bad"

    def play(self, agent: AbsAgent) -> Action:
        return Action.D


class Random(AbsStrategy):
    id = "random"

    def play(self, agent: AbsAgent) -> Action:
        return Action(random.randint(0, 1))


class Tit4Tat(AbsStrategy):
    id = "tit4tat"

    """
    This is a tit4tat strategy that bases the tit4tat on their last interaction with that agent
    """
    def play(self, agent: AbsAgent) -> Action:
        if agent.env.rounds <= 1:
            return Action.C
        if agent.memory and agent.memory[agent.opponent][1] == 1:
            return Action.D
        return Action.C


class Tit4Tat2(AbsStrategy):
    id = "tit4tat2"

    """
    This tit4tat uses the opposing agent last move to determine
    whether it is going to defect or not
    """
    def play(self, agent: AbsAgent) -> Action:
        if agent.opponent is None:
            return Action.D
        opp_last = agent.env.agents[agent.opponent].last_action
        if opp_last is None:
            return Action.D
        if opp_last == 1:
            return Action.C
        return Action.D


class RevTit4Tat(AbsStrategy):
    id = "rev_tit4tat"

    """
    This is a reverse tit4tat strategy that bases the tit4tat on their last interaction with that agent
    """
    def play(self, agent: AbsAgent) -> Action:
        if agent.env.rounds <= 1:
            return Action.C
        if agent.memory[agent.opponent][1] == 1:
            return Action.D
        return Action.C


class RevTit4Tat2(AbsStrategy):
    id = "rev_tit4tat2"

    """
    This reverse tit4tat uses the opposing agent last move to determine
    whether it is going to defect or not
    """
    def play(self, agent: AbsAgent) -> Action:
        if agent.opponent is None:
            return Action.D
        opp_last = agent.env.agents[agent.opponent].last_action
        if opp_last is None:
            return Action.D
        if opp_last == 1:
            return Action.C
        return Action.D


class WinStayLoseShift(AbsStrategy):
    id = "win_stay_lose_shift"
    """
    The win stay lose shift strategy takes the outcome of the previous play
    and divided into success (wins) and failures (losses).
    The agent plays the same strategy on the next round if previous results in success. 
    Alternatively, if the play resulted in a failure the agent switches to another action. 
    ----------------------------
    Here consider last reward >= last reward got for opponent as a success
    """
    def play(self, agent: AbsAgent) -> Action:
        if agent.last_opponent is None:
            return Action.C
        last_reward_opp = agent.env.agents[agent.last_opponent].last_reward
        if agent.last_reward >= last_reward_opp: # can choose defection or cooperation
            return agent.last_action
        return random.choice(list(filter(lambda x: x != agent.last_action, list(Action))))


registry.add(StaticAgent, Good)
registry.add(StaticAgent, Bad)
registry.add(StaticAgent, Random)
registry.add(StaticAgent, Tit4Tat)
registry.add(StaticAgent, Tit4Tat2)
registry.add(StaticAgent, RevTit4Tat)
registry.add(StaticAgent, RevTit4Tat2)
registry.add(StaticAgent, WinStayLoseShift)
