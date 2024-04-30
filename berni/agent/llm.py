from nypd.environment import BaseEnv
from nypd.agent import BaseAgent
from nypd.strategy import AbsStrategy
from nypd.agent.registry import agent_registry


class LLMAgent(BaseAgent):

    name = "llm"

    def __init__(
            self,
            env: BaseEnv,
            id: int,
            rules_prompt: str,
            system_prompt: str,
            strategy: AbsStrategy | None = None,
            norm_weight: float = 1
    ):
        super().__init__(env=env, id=id, strategy=strategy, norm_weight=norm_weight)

        self.rules_prompt = rules_prompt
        self.system_prompt = system_prompt

        self.quiz_performance = {}

    def act(self, opponent: BaseAgent | None = None):
        self.opponent = opponent
        if self.strategy is None:
            raise ValueError("Strategy is not defined")
        action = self.strategy.play(agent=self, opponent=self._env.agents[opponent])
        self.action = action
        return action

    def preplay(self, opponent: BaseAgent | None = None):
        self.opponent = opponent
        if self.strategy is None:
            raise ValueError("Strategy is not defined")
        return self.strategy.preplay(agent=self, opponent=self._env.agents[opponent])
    
    def postplay(self, output: str, opponent: BaseAgent | None = None):
        self.opponent = opponent
        if self.strategy is None:
            raise ValueError("Strategy is not defined")
        action = self.strategy.postplay(agent=self, opponent=self._env.agents[opponent], output=output)
        self.action = action
        return action


agent_registry.add(LLMAgent)
