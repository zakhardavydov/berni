from typing import Dict, Optional, List

from ..structures.action import Action
from ..structures.agent_history_item import AgentHistoryItem

from .abstract import AbsAgent


class BaseAgent(AbsAgent):

    name = "base"

    def __init__(
            self,
            env,
            id: int,
            strategy=None,
            norm_weight=1
    ):

        self._env = env
        self.id = id
        self._strategy = strategy
        self._norm_weight = norm_weight

        self._memory: Dict[AbsAgent, List[Action]] = {}
        self._history: List[AgentHistoryItem] = []

        self._opponent: Optional[AbsAgent] = None
        self.opponent_model: Optional[AbsAgent] = None

        self._score = 0

        self._action: Optional[Action] = None

    def clean(self):
        self._score = 0
        self._opponent = None
        self._action = None
        self._history = []
        self._memory = {}

    def act(self, opponent: Optional['BaseAgent'] = None):
        """
        Dependent on whether a strategy is given it takes that structures
        or applies a random structures
        :opponent - gets the current opponent
        """
        self.opponent = opponent
        if self.strategy is not None:
            self.action = self.strategy.play(agent=self, opponent=self._env.agents[opponent])
        else:
            raise ValueError("Strategy is not defined")
        return self.action

    @property
    def env(self):
        return self._env
    
    @property
    def norm_weight(self):
        return self._norm_weight

    @env.setter
    def env(self, value):
        self._env = value
    
    @norm_weight.setter
    def norm_weight(self, value):
        self._norm_weight = value

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, value):
        self._action = value

    @property
    def memory(self) -> Dict[AbsAgent, List[Action]]:
        return self._memory

    @memory.setter
    def memory(self, value):
        self._memory = value

    @property
    def opponent(self) -> Optional[AbsAgent]:
        return self._opponent

    @opponent.setter
    def opponent(self, value: Optional[AbsAgent]):
        self._opponent = value

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        self._strategy = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @property
    def last_action(self):
        if self._history:
            return self._history[-1].action
        return None

    @property
    def last_reward(self):
        if self._history:
            return self._history[-1].reward
        return None

    @property
    def last_opponent(self):
        if self._history:
            return self._history[-1].opponent
        return None

    @property
    def last_round(self):
        if self._history:
            return self._history[-1].round
        return None

    def get_last_history(self, buffer_size: int = 1) -> List[AgentHistoryItem]:
        return self._history[-buffer_size:]

    def save_history(self, opponent, opponent_model, action, reward, round):
        self.opponent_model = opponent_model
        self._history.append(
            AgentHistoryItem(
                opponent=opponent,
                action=action,
                reward=reward,
                round=round
            )
        )

    def observe(self, reward, state):
        """
        Takes in the state space
        as of right now only the reward is implemented
        The observe function could be used to get the structures of
        """
        self.score += reward
        for act in state:
            self.memory[act[0]] = act[1]
