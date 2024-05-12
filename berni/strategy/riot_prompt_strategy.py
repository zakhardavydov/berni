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
            You are talking with a new neighbour.
            They think:
            Quote starts:
            "{opponent_opinion}"
            Quote ends.
            [INST]
            State in one word whether you agree (yes) or disagree (no) with the neighbour and summarise your new opinion.
            Template for response: "yes/no, opinion"
            Response:
            [/INST]
            """

    def preplay(self, agent: RiotLLMAgent, opponent: RiotLLMAgent) -> str | None:
        prompt = self._prompt_builder(agent, opponent)
        agent.round_prompt = prompt
        if agent.bias_score == opponent.bias_score or agent.opinion == opponent.opinion:
            return None
        return prompt
    
    def postplay(self, agent: RiotLLMAgent, opponent: RiotLLMAgent, output: str) -> Action:
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

    def play(self, agent: RiotLLMAgent, opponent: RiotLLMAgent) -> Action:
        prompt = self.preplay(agent, opponent)
        if not prompt:
            return Action.D
        with llm_call_lock:
            output = self._llm.predict(agent.round_prompt)
        return self.postplay(agent, opponent, output)


registry.add(RiotLLMAgent, RiotPromptStrategy)
