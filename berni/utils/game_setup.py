from pydantic import BaseModel

from .llm_builder import BerniLLMConfig


class BerniGameSetup(BaseModel):
    game: str
    ps: str
    results_repo: str
    agent_name: str
    bias_primer: list[float]
    llm: BerniLLMConfig
    grid_size: list[int]
    num_rounds: int
    seeds: list[int]
    setup_name: str | None = None
