import numpy as np

from pydantic import BaseModel

from nypd.agent import AbsAgent, BaseAgent, agent_registry,
from nypd.ps import AbsPartnerSelection
from nypd.seed import AbsSeed, NaiveSeed
from nypd.structures import AgentConfigs
from nypd.env import AbsEnv


class GridNeighbourPartnerSelection(AbsSeed, AbsPartnerSelection):
    
    def __init__(self, grid_size: int, parties: AgentConfigs):
        self.grid_size = grid_size
        self.cell_size = grid_size * grid_size

        self._party = None
        self._party_index = None
        self._party_prob = None
        self._matrix = None

    def generate_grid(n, m, object_types, probabilities):
        if len(object_types) != len(probabilities):
            raise ValueError("Lengths of object_types and probabilities must be the same.")

        if not np.isclose(np.sum(probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1.")

        rand_nums = np.random.rand(n, m)

        grid = np.zeros((n, m), dtype=int)

        for obj_type, prob in zip(object_types, probabilities):
            grid[rand_nums < prob] = obj_type

        return grid
    
    def seed(
            self,
            env: AbsEnv,
            agents: AgentConfigs,
            count: int
    )  -> tuple[list[BaseAgent], dict[str, float]]:
        self._party = {i: party for i, party in enumerate(agents.configs)}
        self._party_index = [i for i, _ in enumerate(agents.configs)]
        self._party_prob = [p.ratio for p in agents.configs]
        self._matrix = self.generate_grid(self.grid_size, self.grid_size, self._party_index, self._party_prob)
        
        out = []

        st_count = NaiveSeed.get_st_count()

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                party = self._party_index[self._matrix[row, col]]
                constructor = agent_registry.registry[party.config]
                strategy, st_id = NaiveSeed.pick_strategy(party, constructor)
                agent_id = row * self.grid_size + col
                agent = constructor(env=env, id=agent_id, strategy=strategy, **party.params)
                out.append(agent)
                st_count[st_id] += 1
        st_ratio = {key: value / count for key, value in st_count.items()}
        return out, st_ratio
    
    def select(prev: list[list[AbsAgent]] | None, round: int, num_agents: int) -> list[list[AbsAgent]]:
        ...
