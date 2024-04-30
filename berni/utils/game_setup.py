from typing import Any

from pydantic import BaseModel

from .llm_builder import BerniLLMConfig


class BerniPSConfig(BaseModel):
    name: str
    params: dict[str, Any] | None = None


class BerniGameSetup(BaseModel):
    game: str
    prompts: list[str]
    ps: BerniPSConfig
    results_repo: str
    agent_name: str
    bias_primer: list[float]
    llm: BerniLLMConfig
    grid_size: list[int]
    num_rounds: int
    seeds: list[int]
    controlled_agent: float
    setup_name: str | None = None
