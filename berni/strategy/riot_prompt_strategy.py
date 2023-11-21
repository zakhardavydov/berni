from pydantic import BaseModel, Field
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser

from nypd.strategy import AbsStrategy
from nypd.structures import Action

from berni.agent import RiotLLMAgent


class RiotPromptStrategy(AbsStrategy):

    def __init__(self, llm: BaseLanguageModel, id: str, prompt: str):
        self.id = id

        self._prompt = prompt
        self._llm = llm

    def _prompt_builder(self, agent: RiotLLMAgent, opponent: RiotLLMAgent) -> str:
        opponent_opinion = opponent.opinion
        return f"""
            {agent.rules_prompt}\n
            {agent.system_prompt}\n
            {self._prompt}\n
            You are talking to a neighbour in a crowd. Quote from their speech: "{opponent_opinion}". Do you think their behavior is a better way to convey your point?
            Respond in one word: "agree" or "disagree".
            """

    def play(self, agent: RiotLLMAgent, opponent: RiotLLMAgent) -> Action:
        prompt = self._prompt_builder(agent, opponent)
        print("-------------")
        print(prompt)
        output = self._llm.predict(prompt)
        print("-----")
        print(output)
        if "disagree" in output.lower():
            return Action.D
        agent.opinion = opponent.opinion
        return Action.C
