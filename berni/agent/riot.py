from nypd.env import BaseEnv
from nypd.agent import BaseAgent
from nypd.strategy import AbsStrategy
from nypd.agent.registry import agent_registry

from .llm import LLMAgent


class RiotLLMAgent(LLMAgent):

    name = "opinion_llm"

    def __init__(
            self,
            env: BaseEnv,
            id: int,
            rules_prompt: str,
            system_prompt: str,
            strategy: AbsStrategy | None = None,
            norm_weight: float = 1,
            bias_score: float | None = None
    ):
        super().__init__(
            env=env,
            id=id,
            rules_prompt=rules_prompt,
            system_prompt=system_prompt,
            strategy=strategy,
            norm_weight=norm_weight
        )
        
        self.opinion = self.system_prompt
        self.initial_bias = bias_score
        self.bias_score = bias_score

        self.round_prompt = None


agent_registry.add(RiotLLMAgent)
