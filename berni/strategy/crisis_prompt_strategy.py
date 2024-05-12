import threading

from pydantic import BaseModel, Field
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser

from nypd.strategy import AbsStrategy, registry
from nypd.structures import Action

from berni.agent import LLMAgent, RiotLLMAgent

from .prompt_strategy import PromptStrategy


llm_call_lock = threading.Lock()


class CrisisPromptStrategy(PromptStrategy):

    name = "crisis"

    def __init__(self, llm: BaseLanguageModel, id: str, prompt: str, debate_topic: str):
        self.debate_topic = debate_topic
        super().__init__(llm, id, prompt)
        
    def base_prompt(self, agent: RiotLLMAgent) -> str:
        return f"{agent.rules_prompt}\n{agent.opinion}."

    def _prompt_builder(self, agent: RiotLLMAgent, opponent: RiotLLMAgent) -> str:
        opponent_action = opponent.round_prompt.lower()
        if opponent_action:
            return f"""
                {self.base_prompt(agent)}
                You are dealing with a request to support a new action.
                "{opponent_action}"
                [INST]
                State in one word whether you support the action (yes) or decline (no). Next, state your next action.
                Template for response: "yes/no; action"
                Response:
                [/INST]
                """
        return f"""
            {self.base_prompt(agent)}
            [INST]
            You are acting to address the crisis.
            In one sentence, state your next response action:
            [/INST]
            """

    def preplay(self, agent: RiotLLMAgent, opponent: RiotLLMAgent) -> str | None:
        prompt = self._prompt_builder(agent, opponent)
        return prompt
    
    def postplay(self, agent: RiotLLMAgent, opponent: RiotLLMAgent, output: str) -> Action:
        output = output.split(";")
        if output:
            is_cooperate = "yes" in output[0].lower()
            if is_cooperate:
                agent.bias_score = (agent.bias_score + opponent.bias_score) / 2
                return Action.C
            agent.round_prompt = output
            return Action.D
        agent.round_prompt = output
        return Action.D

    def play(self, agent: RiotLLMAgent, opponent: RiotLLMAgent) -> Action:
        prompt = self.preplay(agent, opponent)
        with llm_call_lock:
            output = self._llm.predict(prompt)
        return self.postplay(agent, opponent, output)


registry.add(RiotLLMAgent, CrisisPromptStrategy)
