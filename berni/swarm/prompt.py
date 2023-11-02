from langchain.llms.base import BaseLanguageModel

from nypd.structures import AgentConfigs, AgentConfigPrePlay, AgentConfig

from berni.strategy import PromptStrategy

from .abs import AbsSwarmGenerator


class PromptSwarmGenerator(AbsSwarmGenerator):

    def __init__(
            self,
            llm: BaseLanguageModel,
            agent_prompt_dir: str,
            strategy_prompt_dir: str,
            rule_prompt_path: str,
            default_agent_type: str = "llm"
    ):
        self._llm = llm
        self._default_agent_type = default_agent_type

        self.agents = self.get_prompts(agent_prompt_dir)
        self.strategies = self.get_prompts(strategy_prompt_dir)
        self.rules_prompt = self.get_prompt(rule_prompt_path)

    def get_agent_ratio(self) -> dict[str, float]:
        return {
            agent_name: 1 / len(self.agents) for agent_name in self.agents.keys()
        }

    def get_strategy_ratio(self) -> dict[str, float]:
        return {
            strategy_name: 1 / len(self.strategies) for strategy_name in self.strategies.keys()
        }

    def _init_strategy(self) -> list[PromptStrategy]:
        return [
            PromptStrategy(llm=self._llm, id=strategy_name, prompt=prompt)
            for strategy_name, prompt in self.strategies.items()
        ]

    def generate(self, *args, **kwargs) -> tuple[AgentConfigs, list[PromptStrategy]]:
        ratio = self.get_agent_ratio()
        strategy = self._init_strategy()
        configs = []
        for agent_name, prompt in self.agents.items():
            pre_play = AgentConfigPrePlay(
                config=AgentConfig(
                    name=agent_name,
                    type=self._default_agent_type,
                    strategy=self.get_strategy_ratio(),
                    params={
                        "system_prompt": prompt,
                        "rules_prompt": self.rules_prompt
                    }
                ),
                ratio=ratio[agent_name]
            )
            configs.append(pre_play)
        return AgentConfigs(configs=configs), strategy
