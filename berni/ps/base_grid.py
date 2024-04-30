from abc import ABC, abstractmethod

import numpy as np

from pydantic import BaseModel

from nypd.agent import AbsAgent, BaseAgent, agent_registry
from nypd.seed import AbsSeed, NaiveSeed
from nypd.structures import AgentConfigs
from nypd.environment import AbsEnv

from .structured import StructuredPartnerSelection


class BaseGridPartnerSelection(AbsSeed, StructuredPartnerSelection, ABC):
    
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.cell_size = grid_size * grid_size

        self._party = None
        self._party_index = None
        self._party_prob = None
        self._matrix = None

    def num_agents(self) -> int:
        return self.cell_size

    def generate_grid(self, n, m, object_types, probabilities):
        shape = (n, m)
        grid = np.empty(shape, dtype=int)
        flat_shape = np.prod(shape)
        
        # Flatten probabilities and normalize to sum to 1
        normalized_probs = np.array(probabilities) / np.sum(probabilities)
        
        # Generate random values based on probabilities
        choices = np.random.choice(np.arange(len(object_types)), size=flat_shape, p=normalized_probs)
        
        # Reshape the choices to match the grid shape
        grid = choices.reshape(shape)
        return grid
    
    def find_row_col(self, num: int) -> tuple[int, int]:
        row = num // self.grid_size 
        col = num % self.grid_size
        return row, col
    
    def row_col_to_index(self, row: int, col: int) -> int:
        return row * self.grid_size + col
    
    def neighbours(self, agent_index: int, depth: int = 1) -> list[int]:
        neighbors = set()
        rows, cols = self._matrix.shape
        visited = set()
        index = self.find_row_col(agent_index)
        
        def is_valid(i, j):
            return 0 <= i < rows and 0 <= j < cols

        def dfs(i, j, cur_depth):
            if cur_depth > depth:
                return
            for x in range(max(0, i - 1), min(i + 2, rows)):
                for y in range(max(0, j - 1), min(j + 2, cols)):
                    if is_valid(x, y) and (x != i or y != j):
                        if (x, y) not in visited:
                            visited.add((x, y))
                            neighbors.add((x, y))
                            if cur_depth + 1 <= depth and self._matrix[x][y]:
                                dfs(x, y, cur_depth + 1)

        i, j = index
        dfs(i, j, 1)
        out = [self.row_col_to_index(neighbour[0], neighbour[1]) for neighbour in list(neighbors)]
        return out
    
    def seed(
            self,
            env: AbsEnv,
            agents: AgentConfigs
    )  -> tuple[list[BaseAgent], dict[str, float]]:
        self._party = {i: party for i, party in enumerate(agents.configs)}
        self._party_index = [i for i, _ in enumerate(agents.configs)]
        print("Agents:\n", self._party)
        self._party_prob = [p.ratio for p in agents.configs]
        print("Agent occurance probabilities:\n", self._party_prob)
        self._matrix = self.generate_grid(self.grid_size, self.grid_size, self._party_index, self._party_prob)
        print("Resultant grid matrix:\n", self._matrix)
        
        out = []

        st_count = NaiveSeed.get_st_count()

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                party = self._party[self._matrix[row, col]]
                constructor = agent_registry.registry[party.config.type]
                strategy, st_id = NaiveSeed.pick_strategy(party.config, constructor)
                agent_id = self.row_col_to_index(row, col)
                agent = constructor(env=env, id=agent_id, strategy=strategy, **party.config.params)
                out.append(agent)
                st_count[st_id] += 1
        st_ratio = {key: value / self.cell_size for key, value in st_count.items()}
        return out, st_ratio
    
    @abstractmethod
    def select(self, prev: list[list[AbsAgent]] | None, round: int, num_agents: int) -> list[list[AbsAgent]]:
        raise NotImplementedError
