from typing import Optional

from ..structures.action import Action

from .base import BaseAgent
from .registry import agent_registry


class StaticAgent(BaseAgent):
    """
    The agent function takes in one argument:
    strategy - a function that outputs what structures an agent should take
    env - the environment the game is being played in
    id - the id number of the agent (more for debugging purposes)

    Initialises:
    memory - stores all the agent interactions in order with other agent (in this case it is for one round)
    score - the total score of the agent gathered in this iteration
    last_opponent - stores last opponent
    opponent - stores the current opponent
    structures - stores the current structures
    last - stores the last structures
    """

    name = "static"

    def __init__(
            self,
            env,
            id: int,
            strategy=None,
            norm_weight=1
    ):
        super().__init__(env=env, id=id, strategy=strategy, norm_weight=norm_weight)


agent_registry.add(StaticAgent)
