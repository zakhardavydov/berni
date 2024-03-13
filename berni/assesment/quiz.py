import pandas as pd

from tqdm import tqdm

from pydantic import BaseModel
from langchain.llms.base import BaseLanguageModel
from langchain.prompts import PromptTemplate

from nypd.agent import AbsAgent

from berni.agent import LLMAgent
from berni.strategy import PromptStrategy

from .agent import AgentAssesor


class AgentQuestionResponse(BaseModel):
    question: str
    answer: str
    score: int


class AgentResponse(BaseModel):
    score: int
    responses: list[AgentQuestionResponse] | None = None


class AgentLabel(BaseModel):
    label: str


class LLMAgentQuiz(AgentAssesor[AgentResponse, AgentLabel]):

    def __init__(
            self,
            name: str,
            question_path: str,
            prompt_path: str,
            allowed_answers: dict[str, int],
            llm: BaseLanguageModel,
            classes: dict[int, AgentLabel] | None = None,
            specifc_questions: list[int] | None = None
    ) -> None:
        super().__init__()

        _df = pd.read_csv(question_path)

        assert "question" in _df.columns

        self._questions = _df["question"].to_list()

        if specifc_questions:
            self._questions = [self._questions[index] for index in specifc_questions]

        self._name = name
        self._llm = llm
        self._allowed_answers = allowed_answers
        self._classes = classes
        
        with open(prompt_path, "r") as f:
            self._prompt = PromptTemplate.from_template(f.read())

    def assess(self, agent: LLMAgent, strategy: PromptStrategy) -> AgentResponse:
        base_prompt = strategy.base_prompt(agent)
        total = 0
        responses = []
        for question in tqdm(self._questions):
            formatted = self._prompt.format(agent_prompt=base_prompt, question=question, allowed_answers=",".join(list(self._allowed_answers.keys())))
            print("RUNNING QUIZ")
            print(formatted)
            response = self._llm.invoke(formatted).content.replace(".", "").lower()
            print(response)
            assert response in self._allowed_answers, "LLM responded in wrong format"
            score = self._allowed_answers[response]
            total += score
            responses.append(AgentQuestionResponse(question=question, answer=response, score=score))
        performance = AgentResponse(score=total, responses=responses)
        agent.quiz_performance[self._name] = performance
        return performance
    
    def classify(self, response: AgentResponse) -> AgentLabel:
        assert self._classes, "Classes were not provided"
        lower_keys = [key for key in self._classes.keys() if key < response.score]
        assert lower_keys
        closest_key = max(lower_keys)
        return self._classes[closest_key]
