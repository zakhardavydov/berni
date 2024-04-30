import numpy as np
import random as rand

from typing import List, Optional

from nypd.structures import ExperimentSetup
from nypd.ps import AbsPartnerSelection
from nypd.game import game_registry
from nypd.norms import norm_registry

from ..game import AbsGame
from ..structures.action import Action
from ..agent import AbsAgent

from .abstract import AbsEnv
from ..norms import AbsNorm
from ..ps import AbsPartnerSelection


class BaseEnv(AbsEnv):
    """
    The environment class is a prisoner's dilemma:
    Initialised with the number of agent and rounds

    'self.done' is a boolean for whether the game is finished
    'self.game' is the matrix representing the social dilemma
    'self.pairs' is the all combinations of agent games

    The following arrays all have arrays of size (num of agent)
    ____________________________
    'self.agent' is the array that holds the Agent Objects
    'self.scores' tracks the total scores for the number of rounds
    'self.rounds' tracks the current round number
    'self.structures' tracks the current round structures for each agent
    'self.rewards' tracks the rewards of each agent
    _____________________________

    """

    def __init__(self, num_agents: int, num_rounds: int, game: AbsGame, norm: AbsNorm, ps: AbsPartnerSelection):
        self.num_agents = num_agents
        self._num_rounds = num_rounds
        self.game = game
        self.norm = norm
        self.ps = ps

        self.agents = []
        self._actions: List[Action] = []
        self._scores = np.zeros(num_agents)
        self._prev_round_scores: List[float] = []
        self._gini_coefficient = 1
        self._current_payoffs = dict()
        self._rewards = np.zeros(self.num_agents)
        self.state = self.__empty_state()

        self.rounds = 0
        self.done = False

    def setup(self):
        self.pairs = self.ps.select(None, 0, self.num_agents)

    def complete(self):
        pass

    @property
    def scores(self):
        return self._scores
    
    @scores.setter
    def scores(self, value):
        self._scores = value

    @property
    def prev_round_scores(self):
        return self._prev_round_scores
    
    @prev_round_scores.setter
    def prev_round_scores(self, value):
        self._prev_round_scores = value

    @property
    def current_payoffs(self):
        return self._current_payoffs
    
    @current_payoffs.setter
    def current_payoffs(self, value):
        self._current_payoffs = value

    @property
    def gini_coefficient(self):
        return self._gini_coefficient
    
    @gini_coefficient.setter
    def gini_coefficient(self, value):
        self._gini_coefficient = value

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, value):
        self._actions = value

    @property
    def rewards(self):
        return self._rewards

    @rewards.setter
    def rewards(self, value):
        self._rewards = value

    @property
    def num_rounds(self):
        return self._num_rounds

    @num_rounds.setter
    def num_rounds(self, value):
        self._num_rounds = value

    def add_agent(self, agent):
        """
        Takes in the Agent object and stores it in the environment
        """
        self.agents.append(agent)

    def reset(self):
        """
        reset() - resets the game back to square 1
        """
        self.actions = []
        self.scores = np.zeros(self.num_agents)
        self.rewards = np.zeros(self.num_agents)

        self.rounds = 0
        self.done = False

    def __empty_state(self):
        return [[] for _ in range(self.num_agents)]

    def step(self):
        """
        step() calculates the structures of all the agent interactions

        All agent play against each other and each agent interaction will have a different
        structures in accordance with the strategy that agent is using

        The agent are then updated with the rewards and memory of the current state

        ###
        This is where partner selection would have to be done
        """
        self.rounds += 1
        # Removing structures and rewards of the previous round
        self.actions = []
        self.rewards = np.zeros(self.num_agents)
        self.state = self.__empty_state()

        self.pairs = self.ps.select(self.pairs, 0, self.num_agents)
        self.prev_round_scores = self.scores.copy()
        
        for i, j in self.pairs:
            agent1: AbsAgent = self.agents[i]
            agent2: AbsAgent = self.agents[j]

            bias_gap = agent2.bias_score - agent1.bias_score

            # Checking the reward matrix and assigning it to the respective agent reward
            act1 = agent1.act(j)
            act2 = agent2.act(i)

            payoff = self.game.get_payoff(action=(act1, act2))
            self.current_payoffs.clear()
            self.current_payoffs[i] = payoff[0]
            self.current_payoffs[j] = payoff[1]

            w1 = self.norm.calculate_reward(agent1, agent2)
            w2 = self.norm.calculate_reward(agent2, agent1)

            r1 = payoff[0] + w1
            r2 = payoff[1] + w2

            self.actions.append((act1, act2))

            self.state[i].append((j, (act1, act2)))
            self.state[j].append((i, (act2, act1)))

            self.rewards[i] += r1
            self.rewards[j] += r2

            agent1.save_history(
                opponent=j,
                opponent_model=agent1,
                action=act1,
                reward=r1,
                round=self.rounds,
                bias_gap=bias_gap
            )
            agent2.save_history(
                opponent=i,
                opponent_model=agent2,
                action=act2,
                reward=r2,
                round=self.rounds,
                bias_gap=bias_gap
            )

        # Updating the agent rewards in their objects
        for i in range(self.num_agents):
            self.agents[i].observe(self.rewards[i], self.state[i])

        # Updating total score and checking if the game is done
        self.scores += self.rewards

        if self.rounds >= self.num_rounds:
            self.done = True

    @staticmethod
    def get_game_norm(exp_setup: ExperimentSetup) -> tuple[AbsGame, AbsNorm]:
        game = game_registry.registry[exp_setup.game](**exp_setup.game_params)
        norm = norm_registry.registry[exp_setup.norm](**exp_setup.norm_params)
        return game, norm

    @staticmethod
    def from_exp_setup(exp_setup: ExperimentSetup, ps: AbsPartnerSelection) -> 'BaseEnv':
        game, norm = BaseEnv.get_game_norm(exp_setup)
        return BaseEnv(num_agents=exp_setup.num_agents, num_rounds=exp_setup.num_rounds, game=game, norm=norm, ps=ps)
