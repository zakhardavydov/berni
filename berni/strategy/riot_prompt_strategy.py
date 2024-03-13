import threading

from pydantic import BaseModel, Field
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser

from nypd.strategy import AbsStrategy
from nypd.structures import Action

from berni.agent import LLMAgent, RiotLLMAgent

from .prompt_strategy import PromptStrategy


llm_call_lock = threading.Lock()


class RiotPromptStrategy(PromptStrategy):

    def __init__(self, llm: BaseLanguageModel, id: str, prompt: str):
        super().__init__(llm, id, prompt)
        
    def base_prompt(self, agent: RiotLLMAgent) -> str:
        return f"{agent.rules_prompt}\n{self._prompt}\nYour opinion: Quote starts: {agent.opinion} Quote ends."

    def _prompt_builder(self, agent: RiotLLMAgent, opponent: RiotLLMAgent) -> str:
        opponent_opinion = opponent.opinion.lower()
        return f"""
            {self.base_prompt(agent)}
            You are talking with a neighbour.
            They think:
            Quote starts:
            "{opponent_opinion}"
            Quote ends.
            In one line, do you agree?
            Clearly state yes/no:
            """

    def play(self, agent: RiotLLMAgent, opponent: RiotLLMAgent) -> Action:
        prompt = self._prompt_builder(agent, opponent)
        agent.round_prompt = prompt
        print("-------------")
        print(prompt)
        if agent.bias_score == opponent.bias_score or agent.opinion == opponent.opinion:
            return Action.D
        with llm_call_lock:
            output = self._llm.predict(prompt)
        print("-----")
        print(output)
        output = output.split(",")
        answer = output[0]
        opinion = " ".join(output[1:])
        try:
            if "no" in answer.lower():
                return Action.D
            agent.opinion = opinion
            if agent.opinion == "":
                agent.opinion = opponent.opinion
            agent.bias_score = opponent.bias_score
            return Action.C
        except Exception as e:
            print(e)
        return Action.D
