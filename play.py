import os
import uuid
import time
import yaml
import random
import argparse
import threading

import numpy as np
import pandas as pd

from tqdm import tqdm

from langchain.llms.base import BaseLanguageModel

from nypd.core import settings
from nypd.structures import ExperimentSetup
from nypd.strategy import registry

from berni.agent import RiotLLMAgent
from berni.swarm import PromptSwarmGenerator
from berni.ps import RandomGridPartnerSelection
from berni.environment import LLMEnv

from berni.utils import llm, BerniGameSetup
from berni.core import settings


def bias_generator(primer: float, primer_bias: float, non_primer: list[float]) -> dict[float: float]:
    out = {primer_bias: primer}
    non_primer_ratio = (1 - primer) / len(non_primer)
    for non in non_primer:
        out[non] = non_primer_ratio
    return out


def run_simulation(
        setup_name: str,
        used_llm: BaseLanguageModel,
        agent_name: str,
        game: str,
        ps: str,
        batch_id: int,
        grid_size: int,
        num_rounds: int,
        seed: int,
        save_dir: str = "metrics",
        bias_ratio_override: dict[float, float] | None = None
):
    
    np.random.seed(seed)
    random.seed(int(seed))

    internal_sim_id = str(uuid.uuid4())

    prompt = os.path.join(settings.PROMPT_DIR, game)

    active_ps = None
    if ps == "random_grid":
        active_ps = RandomGridPartnerSelection(grid_size=grid_size)
        
    assert active_ps, "Selected partner selection failed to parse"

    bias_scores = pd.read_csv(os.path.join(prompt, "radical_scoring.csv"))
    bias_scores = {row["agent"]: row["radical"] for _, row in bias_scores.iterrows()}

    print(bias_scores)
    
    generator = PromptSwarmGenerator(
        llm=used_llm,
        agent_prompt_dir=os.path.join(prompt, settings.AGENT_PROMPT_DIR),
        strategy_prompt_dir=os.path.join(prompt, settings.STRATEGY_PROMPT_DIR),
        rule_prompt_path=os.path.join(prompt, settings.RULES_PROMPT_PATH),
        bias_scores=bias_scores,
        default_agent_type=agent_name
    )

    agent_config, strategy = generator.generate(ratio_override=bias_ratio_override)

    generator.register_strategy(registry, [(RiotLLMAgent, s) for s in strategy])

    exp_setup = ExperimentSetup(
        num_agents=active_ps.num_agents(),
        num_rounds=num_rounds,
        game=settings.DEFAULT_MATRIX_GAME,
        norm=settings.DEFAULT_NORM,
        config=agent_config
    )

    report_dir = os.path.join(save_dir, batch_id)
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    env = LLMEnv.from_exp_setup(exp_setup, active_ps)
    env.results_dir = os.path.join(report_dir, internal_sim_id)

    agents, _ = active_ps.seed(env=env, agents=exp_setup.config)

    env.reset()

    for agent in agents:
        agent.clean()
        env.add_agent(agent)
    
    env.setup()

    for _ in tqdm(range(env.num_rounds)):
        env.step()

    env.complete()

    metrics = []
    for i in range(0, env.num_rounds):
        metric = {
            "sim_id": internal_sim_id,
            "batch_id": batch_id,
            "grid_size": grid_size,
            "num_rounds": num_rounds,
            "round": i,
            "bias": env.bias[i],
            "bias_av": env.bias_av[i],
        }

        for ac in agent_config.configs:
            metric[f"opinion_group__{ac.config.params.get('bias_score')}"] =  ac.ratio

        metrics.append(metric)
    
    metric_df = pd.DataFrame.from_records(metrics)

    report_path = os.path.join(report_dir, f"{internal_sim_id}.csv")

    metric_df.to_csv(report_path)


def run_simulation_threaded(setup_name, game, agent_name, ps, seed, used_llm, batch_id, grid_size, num_rounds, save_dir, bias_ratio_override):
    run_simulation(
        setup_name=setup_name,
        game=game,
        ps=ps,
        seed=seed,
        used_llm=used_llm,
        agent_name=agent_name,
        batch_id=batch_id,
        grid_size=grid_size,
        num_rounds=num_rounds,
        save_dir=save_dir,
        bias_ratio_override=bias_ratio_override
    )


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


def run_simulations(setup_path: str):
    setups = load_setups(setup_path)
    for setup in setups:
        used_llm = llm(setup.llm)
        threads = []
        results = f"{setup.results_repo}/{setup.setup_name}--{str(uuid.uuid4())}"
        for size in setup.grid_size:
            internal_batch_id = str(uuid.uuid4())
            for bias in setup.bias_primer:
                for seed in setup.seeds:
                    bias_ratio_override = bias_generator(bias, -1, [0, 1])
                    t = threading.Thread(target=run_simulation_threaded, args=(
                        setup.setup_name,
                        setup.game,
                        setup.agent_name,
                        setup.ps,
                        seed,
                        used_llm,
                        internal_batch_id,
                        size,
                        setup.num_rounds,
                        results,
                        bias_ratio_override,
                    ))
                    threads.append(t)
                    t.start()

                    time.sleep(1)

        for t in threads:
            t.join()

    print("All simulations completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Run simulation',
        description='Run LLM BERNI simulation'
    )
    parser.add_argument("path")

    args = parser.parse_args()

    run_simulations(args.path)
