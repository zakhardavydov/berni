import threading

from pydantic import BaseModel, Field
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser

from nypd.strategy import AbsStrategy, registry
from nypd.structures import Action

from berni.agent import LLMAgent, RiotLLMAgent

from .prompt_strategy import PromptStrategy


llm_call_lock = threading.Lock()


class RiotPromptStrategy(PromptStrategy):

    name = "riot"

    def __init__(self, llm: BaseLanguageModel, id: str, prompt: str, debate_topic: str):
        self.debate_topic = debate_topic
        super().__init__(llm, id, prompt)
        
    def base_prompt(self, agent: RiotLLMAgent) -> str:
        return f"{agent.rules_prompt}\n{self._prompt}\nYour opinion: Quote starts: {agent.opinion} Quote ends."

    def _prompt_builder(self, agent: RiotLLMAgent, opponent: RiotLLMAgent) -> str:
        opponent_opinion = opponent.opinion.lower()
        return f"""
            {self.base_prompt(agent)}
            You are debating with a neighbour. One of you is wrong.
            They just said:
            Quote starts:
            "{opponent_opinion}"
            Quote ends.

            Respond:
            """

    def preplay(self, agent: RiotLLMAgent, opponent: RiotLLMAgent) -> str | None:
        prompt = self._prompt_builder(agent, opponent)
        agent.round_prompt = prompt
        if agent.bias_score == opponent.bias_score or agent.opinion == opponent.opinion:
            return None
        return prompt
    
    def postplay(self, agent: RiotLLMAgent, opponent: RiotLLMAgent, output: str) -> Action:
        print(output)
        agent.opinion = output
        return Action.C

    def play(self, agent: RiotLLMAgent, opponent: RiotLLMAgent) -> Action:
        prompt = self.preplay(agent, opponent)
        if not prompt:
            return Action.D
        with llm_call_lock:
            output = self._llm.predict(agent.round_prompt)
        return self.postplay(agent, opponent, output)


registry.add(RiotLLMAgent, RiotPromptStrategy)
