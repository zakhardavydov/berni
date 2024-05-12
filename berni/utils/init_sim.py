import os
import yaml

import pandas as pd

from nypd.strategy import registry
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


def init_llm_simulation(
        used_llm,
        agent_name: str,
        prompt: str,
        strategy_name: str,
        grid_size: int,
        ps: BerniPSConfig,
        controlled_agent: int,
        controlled_target_bias: float,
        num_rounds: int
) -> LLMEnv:
    
    picked_agent_constructor = agent_registry.registry[agent_name]
    agent_prompt_strategy = registry[agent_name][strategy_name]
    
    active_ps = None
    active_ps_params = ps.params if ps.params else {}
    if ps.name == "random_grid":
        active_ps = RandomGridPartnerSelection(grid_size=grid_size, **active_ps_params)
        
    assert active_ps, "Selected partner selection failed to parse"
    
    ratio_override = None

    bias_scores = pd.read_csv(os.path.join(prompt, "scoring.csv"))
    bias_scores = {row["agent"]: row["radical"] for _, row in bias_scores.iterrows()}

    non_controlled_agents = [val for val in bias_scores.values() if val != controlled_agent]
    ratio_override = bias_generator(controlled_target_bias, controlled_agent, non_controlled_agents)
    
    generator = PromptSwarmGenerator(
        llm=used_llm,
        agent_prompt_dir=os.path.join(prompt, settings.AGENT_PROMPT_DIR),
        strategy_prompt_dir=os.path.join(prompt, settings.STRATEGY_PROMPT_DIR),
        rule_prompt_path=os.path.join(prompt, settings.RULES_PROMPT_PATH),
        debate_topic_path=os.path.join(prompt, settings.DEBATE_TOPIC_PATH),
        bias_scores=bias_scores,
        default_agent_type=agent_name,
        strategy_constructor=agent_prompt_strategy
    )

    agent_config, strategy = generator.generate(ratio_override=ratio_override)

    generator.register_strategy(registry, [(picked_agent_constructor, s) for s in strategy])

    exp_setup = ExperimentSetup(
        num_agents=active_ps.num_agents(),
        num_rounds=num_rounds,
        game=settings.DEFAULT_MATRIX_GAME,
        norm=settings.DEFAULT_NORM,
        config=agent_config
    )

    env = LLMEnv.from_exp_setup(exp_setup, active_ps)

    agents, _ = active_ps.seed(env=env, agents=exp_setup.config)

    env.reset()

    for agent in agents:
        agent.clean()
        env.add_agent(agent)

    return agent_config, env
