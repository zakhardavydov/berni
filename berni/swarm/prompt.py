from typing import Type

from langchain.llms.base import BaseLanguageModel

from nypd.structures import AgentConfigs, AgentConfigPrePlay, AgentConfig
from nypd.strategy import registry

from berni.strategy import PromptStrategy

from .abs import AbsSwarmGenerator


class PromptSwarmGenerator(AbsSwarmGenerator):

    def __init__(
            self,
            llm: BaseLanguageModel,
            agent_prompt_dir: str,
            rule_prompt_path: str,
            debate_topic_path: str,
            strategy_name: str,
            default_agent_type: str = "llm",
            bias_scores: dict[str, float] | None = None
    ):
        self._llm = llm
        self._default_agent_type = default_agent_type
        self._bias_scores = bias_scores

        self.agents = self.get_prompts(agent_prompt_dir)
        self.rules_prompt = self.get_prompt(rule_prompt_path)
        self.debate_topic = self.get_prompt(debate_topic_path)

        self._strategy_name = strategy_name

    def get_agent_ratio(self) -> dict[str, float]:
        return {
            agent_name: 1 / len(self.agents) for agent_name in self.agents.keys()
        }

    def get_strategy_ratio(self) -> dict[str, float]:
        return {
            self._strategy_name: 1
        }

    def _init_strategy(self) -> list[PromptStrategy]:
        strategy_constructor = registry.registry[self._default_agent_type][self._strategy_name]
        return [
            strategy_constructor(llm=self._llm, id=strategy_constructor.name, prompt="", debate_topic=self.debate_topic)
        ]

    def generate(self, ratio_override: dict[float, float] | None = None, *args, **kwargs) -> tuple[AgentConfigs, list[PromptStrategy]]:
        ratio = self.get_agent_ratio()
        strategy = self._init_strategy()
        configs = []
        for agent_name, prompt in self.agents.items():
            bias = self._bias_scores[agent_name] if self._bias_scores and agent_name in self._bias_scores else None
            pre_play = AgentConfigPrePlay(
                config=AgentConfig(
                    name=agent_name,
                    type=self._default_agent_type,
                    strategy=self.get_strategy_ratio(),
                    params={
                        "system_prompt": prompt,
                        "rules_prompt": self.rules_prompt,
                        "bias_score": bias
                    }
                ),
                ratio=ratio_override[bias] if ratio_override else ratio[agent_name]
            )
            configs.append(pre_play)
        return AgentConfigs(configs=configs), strategy
