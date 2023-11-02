from typing import Dict, Any

from pydantic import BaseModel

from .agent_config import AgentConfigs
import json


class ExperimentSetup(BaseModel):
    num_agents: int
    num_rounds: int
    game: str
    norm: str
    config: AgentConfigs

    game_params: Dict[str, Any] = {}
    norm_params: Dict[str, Any] = {}

    def to_json_dict(self):
        json_dict = {
            "num_agents": self.num_agents,
            "num_rounds": self.num_rounds,
            "game": self.game,
            "norm": self.norm,
            "config": self.config.to_json_dict()
        }

        return json_dict

    def plottable_params(self) -> Dict[str, Any]:
        setup = {
            "num_agents": self.num_agents,
            "num_rounds": self.num_rounds,
            "game": self.game,
            "norm": self.norm,
        }
        for config in self.config.configs:
            setup[f"{config.config.name}__type"] = config.config.type
            setup[f"{config.config.name}__ratio"] = config.ratio
            for param_name, param_value in config.config.params.items():
                setup[f"{config.config.name}__{param_name}"] = param_value
        return setup
