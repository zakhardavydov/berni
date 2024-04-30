import os
import random

from abc import ABC, abstractmethod

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pygraphviz as pgv

from PIL import Image

from nypd import AbsEnv
from nypd.agent import AbsAgent, BaseAgent, agent_registry
from nypd.seed import AbsSeed, NaiveSeed
from nypd.structures.agent_config import AgentConfigs

from .structured import StructuredPartnerSelection


class BaseGraphPartnerSelection(AbsSeed, StructuredPartnerSelection, ABC):

    viz_seed = 42

    def __init__(self, grid_size: int) -> None:
        self._grid_size = grid_size
        self._node_count = grid_size * grid_size

        self._G = None
        self.init_graph()

    def init_graph(self):
        self._G = nx.grid_2d_graph(self._grid_size, self._grid_size)
    
    def neighbours(self, agent_index: int, depth: int = 1) -> list[int]:
        if depth == 1:
            return list(self._G.neighbors(agent_index))
        
        visited = set()
        layer = {agent_index}
        for _ in range(depth):
            next_layer = set()
            for node in layer:
                for neighbor in self._G.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_layer.add(neighbor)
            layer = next_layer
        
        visited.discard(agent_index)
        return list(visited)

    def seed(self, env: AbsEnv, agents: AgentConfigs) -> tuple[list[BaseAgent], dict[str, float]]:
        agent_probabilities = [config.ratio for config in agents.configs]

        strategy_count = {}

        out = []

        for node in range(self._node_count):
            agent_type_index = random.choices(range(len(agent_probabilities)), weights=agent_probabilities, k=1)[0]

            agent_config = agents.configs[agent_type_index]

            constructor = agent_registry.registry[agent_config.config.type]

            strategy, strategy_id = NaiveSeed.pick_strategy(agent_config.config, constructor)

            agent = constructor(env=env, id=node, strategy=strategy, **agent_config.config.params)
            
            out.append(agent)
            
            if strategy_id in strategy_count:
                strategy_count[strategy_id] += 1
            else:
                strategy_count[strategy_id] = 1

            self._G.nodes[node]["agent_type"] = agent_config
            self._G.nodes[node]["strategy"] = strategy

        st_ratio = {key: value / self._node_count for key, value in strategy_count.items()}
        
        return out, st_ratio

    def select(self, prev: list[list[AbsAgent]] | None, round: int, num_agents: int) -> list[tuple[AbsAgent, list[AbsAgent]]]:
        matched = {}
        for agent_index in range(num_agents):
            matched[agent_index] = self.neighbours(agent_index)
        out = [(src, targets) for src, targets in matched.items()]
        return out

    def hash_color(self, value):
        color = plt.cm.coolwarm(0.5 * (value + 1))
        return mcolors.rgb2hex(color)

    def num_agents(self) -> int:
        return len(self._G.nodes)

    def calculate_node_size(self, degree, max_degree):
        min_size = 0.05
        max_size = 0.4
        node_size = min_size + (degree / max_degree) * (max_size - min_size)
        return node_size

    def viz(self, output_path: str):
        base_path = output_path.replace('.png', '_base.png')
        A = pgv.AGraph(strict=True, directed=False)

        degrees = self._G.degree()
        max_degree = max(dict(degrees).values())

        bias_scores = []

        for node, data in self._G.nodes(data=True):
            degree = degrees[node]
            agent_type = data.get('agent_type', 'Unknown')
            bias_score = agent_type.config.params["bias_score"]

            bias_scores.append(bias_score)

            label = f"{bias_score}"

            color = self.hash_color(bias_score)

            node_size = self.calculate_node_size(degree, max_degree)

            A.add_node(node, color="transparent", label="", style='filled', fillcolor=color, shape='circle', width=node_size, height=node_size, fixedsize=True)

        for u, v in self._G.edges():
            A.add_edge(u, v, alpha=0.01, color="#00000060", penwidth=1)

        A.layout(prog='sfdp', args='-Goverlap=scale -Gsplines=true -GK=3')
        A.draw(base_path)

        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)

        cmap = plt.cm.coolwarm
        norm = mcolors.Normalize(vmin=min(bias_scores), vmax=max(bias_scores))

        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
        cb.set_label('Bias Score')

        colorbar_path = base_path.replace('.png', '_colorbar.png')
        plt.savefig(colorbar_path)

        network_img = Image.open(base_path)
        colorbar_img = Image.open(colorbar_path)

        aspect_ratio = colorbar_img.width / colorbar_img.height
        new_height = int(network_img.width / aspect_ratio)

        colorbar_img = colorbar_img.resize((network_img.width, new_height))

        total_height = network_img.height + new_height
        total_height = network_img.height + colorbar_img.height
        combined_img = Image.new('RGBA', (network_img.width, total_height))

        combined_img.paste(network_img, (0, 0))

        combined_img.paste(colorbar_img, (0, network_img.height))

        combined_img.save(output_path)

        os.remove(base_path)
        os.remove(colorbar_path)
