import random

from nypd.agent.qlearning_track_oppo import QLAgentTrackOppon

from ..agent.abstract import AbsAgent
from ..agent.qlearning import QLAgent
from ..structures.action import Action

from .abstract import AbsStrategy
from .registry import registry
import numpy as np


class QTable(AbsStrategy):
    id = "q_table"

    def play(self, agent: QLAgent) -> Action:

        # on first PD, return a random action
        if agent.last_action is None:
            action = random.choice([0, 1])
            return action

        # get a random number to allow the agent to explore possibilities
        if np.random.uniform(0, 1) < agent.exploration_proba:
            action = random.choice([0, 1])
            return action
        
        action = max([0, 1], key=lambda act: agent.qtable.get((agent.internal_state, act), 0.5))

        return action


class QTableTrackOppo(AbsStrategy):
    id = "q_table_tack_oppo"

    def play(self, agent: QLAgent) -> Action:

        # on first PD, return a random action
        if agent.last_action is None:
            action = random.choice([0, 1])
            return action

        # get a random number to allow the agent to explore possibilities
        if np.random.uniform(0, 1) < agent.exploration_proba:
            action = random.choice([0, 1])
            return action
        
        action = max([0, 1], key=lambda act: agent.qtable.get((agent.internal_state, act, agent.opponent), 0.5))

        return action


registry.add(QLAgent, QTable)
# registry.add(QLAgentTrackOppon, QTableTrackOppo) # WIP
