from nypd.env import BaseEnv
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


agent_registry.add(LLMAgent)
