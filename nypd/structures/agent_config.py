from typing import Optional, Union, Dict, Any, List

from pydantic import BaseModel


class AgentConfig(BaseModel):
    type: str
    strategy: Union[str, Optional[Dict[str, float]]]
    params: Dict[str, Any] = {}
    name: Optional[str] = None

    def to_json_dict(self):
        json_dict = {
            "name": self.name,
            "type": self.type,
            "strategy": self.strategy,
            "params": self.params
        }
        return json_dict


class AgentConfigPrePlay(BaseModel):
    config: AgentConfig
    ratio: Optional[float]

    def to_json_dict(self):
        json_dict = {
            "config": self.config.to_json_dict(),
            "ratio": self.ratio
        }
        return json_dict


class AgentConfigs(BaseModel):
    configs: List[AgentConfigPrePlay]

    def to_json_dict(self):
        json_list = []
        for cur_config in self.configs:
            json_list.append(cur_config.to_json_dict())
        
        return json_list
