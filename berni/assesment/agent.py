from typing import TypeVar, Generic

from pydantic import BaseModel

from nypd.agent import AbsAgent


AssResult = TypeVar('AssResult', bound=BaseModel)
Label = TypeVar('Label', bound=BaseModel)


class AgentAssesor(Generic[AssResult, Label]):

    def assess(self, agent: AbsAgent) -> AssResult:
        raise NotImplementedError
    
    def classify(self, agent: AbsAgent) -> Label:
        raise NotImplementedError
