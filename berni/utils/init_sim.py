import os
import yaml

from typing import Any

import pandas as pd

from nypd.strategy import registry, StrategyRegistry
from nypd.agent import agent_registry
from nypd.structures import ExperimentSetup

from berni.core import settings
from berni.ps import *
from berni.swarm import PromptSwarmGenerator
from berni.environment import LLMEnv

from .game_setup import BerniGameSetup, BerniPSConfig


def load_setups(setup_path: str) -> list[BerniGameSetup]:
    setups = []
    with open(setup_path) as stream:
        try:
            loaded = yaml.safe_load(stream)
            for key, config in loaded.items():
                config = BerniGameSetup(**config)
                config.setup_name = key
                setups.append(config)
        except yaml.YAMLError as exc:
            print(exc)
    return setups


def bias_generator(primer: float, primer_bias: float, non_primer: list[float]) -> dict[float: float]:
    out = {primer_bias: primer}
    non_primer_ratio = (1 - primer) / len(non_primer)
    for non in non_primer:
        out[non] = non_primer_ratio
    return out


def topology_constructor(ps: BerniPSConfig, grid_size: int, seed: int) -> AbsSeed:
    constructor = None
    topology_params = ps.params if ps.params else {}
    topology_params["grid_size"] = grid_size
    topology_params["seed"] = seed
    if ps.name == "random_grid":
        constructor = RandomGridPartnerSelection
    elif ps.name == "erdos_renyi":
        constructor = ErdosRenyiPartnerSelection
    elif ps.name == "watts_stogartz":
        constructor = WattsStrogatzPartnerSelection
    elif ps.name == "barbasi_albert":
        constructor = BarabasiAlbertPartnerSelection
    
    assert constructor, "Unknown PS"

    return constructor(**topology_params)


def init_llm_simulation(
        used_llm,
        agent_name: str,
        prompt: str,
        strategy_name: str,
        grid_size: int,
        ps: BerniPSConfig,
        controlled_agent: int,
        controlled_target_bias: float,
        num_rounds: int,
        seed: int
) -> tuple[LLMEnv, AgentConfigs, AbsSeed]:
    
    picked_agent_constructor = agent_registry.registry[agent_name]

    topology = topology_constructor(ps, grid_size, seed)

    ratio_override = None

    bias_scores = pd.read_csv(os.path.join(prompt, "scoring.csv"))
    bias_scores = {row["agent"]: row["radical"] for _, row in bias_scores.iterrows()}

    non_controlled_agents = [val for val in bias_scores.values() if val != controlled_agent]

    ratio_override = bias_generator(controlled_target_bias, controlled_agent, non_controlled_agents)
    
    generator = PromptSwarmGenerator(
        llm=used_llm,
        agent_prompt_dir=os.path.join(prompt, settings.AGENT_PROMPT_DIR),
        rule_prompt_path=os.path.join(prompt, settings.RULES_PROMPT_PATH),
        debate_topic_path=os.path.join(prompt, settings.DEBATE_TOPIC_PATH),
        bias_scores=bias_scores,
        default_agent_type=agent_name,
        strategy_name=strategy_name
    )

    agent_config, strategy = generator.generate(ratio_override=ratio_override)

    temp_registry = StrategyRegistry()
    generator.register_strategy(temp_registry, [(picked_agent_constructor, s) for s in strategy])

    exp_setup = ExperimentSetup(
        num_agents=topology.num_agents(),
        num_rounds=num_rounds,
        game=settings.DEFAULT_MATRIX_GAME,
        norm=settings.DEFAULT_NORM,
        config=agent_config
    )

    env = LLMEnv.from_exp_setup(exp_setup, topology)

    agents, _ = topology.seed(registry=temp_registry, env=env, agents=exp_setup.config)

    env.reset()

    for agent in agents:
        agent.clean()
        env.add_agent(agent)

    return agent_config, env, topology
