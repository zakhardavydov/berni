from pydantic import BaseModel, Field
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser

from nypd.strategy import AbsStrategy
from nypd.structures import Action

from berni.agent import LLMAgent


class PromptResponse(BaseModel):
    justification: str = Field(description="Rationalise step-by-step")
    action: str = Field(description="Single word. Either 'testify' or 'silent'")


class PromptStrategy(AbsStrategy):

    def __init__(self, llm: BaseLanguageModel, id: str, prompt: str):
        self.id = id

        self._prompt = prompt
        self._llm = llm
        self._parser = PydanticOutputParser(pydantic_object=PromptResponse)

    def _prompt_builder(self, agent: LLMAgent) -> str:
        return f"""
            {agent.rules_prompt}\n
            {agent.system_prompt}\n
            {self._prompt}\n
            You can either 'testify' or be 'silent'. In one word, what's your action?:
            """

    def play(self, agent: LLMAgent) -> Action:
        prompt = self._prompt_builder(agent)
        print("-------------")
        print(prompt)
        output = self._llm.predict(prompt)
        print("-----")
        print(output)
        if "silent" in output.lower():
            return Action.C
        return Action.D

