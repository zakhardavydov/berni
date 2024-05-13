import os
import uuid
import time
import yaml
import random
import argparse
import threading
import itertools

import numpy as np
import pandas as pd

from tqdm import tqdm
from langchain.llms.base import BaseLanguageModel

from berni.utils import BerniGameSetup, llm, init_llm_simulation, load_setups, parse_args


def run_simulation(
    experiment_id: str,
    used_llm: BaseLanguageModel,
    batch_id: int,
    grid_size: int,
    prompt: str,
    controlled_target_bias: float,
    seed: int,
    save_dir: str = "metrics",
    setup: BerniGameSetup | None = None
):
    
    np.random.seed(seed)
    random.seed(int(seed))

    internal_sim_id = str(uuid.uuid4())
    report_dir = os.path.join(save_dir, batch_id)

    agent_config, env, topology = init_llm_simulation(
        used_llm,
        setup.agent_name,
        prompt,
        setup.strategy_name,
        grid_size,
        setup.ps,
        setup.controlled_agent,
        controlled_target_bias,
        setup.num_rounds,
        seed
    )
    env.results_dir = os.path.join(report_dir, internal_sim_id)

    env.setup()

    for _ in tqdm(range(env.num_rounds)):
        env.step()

    env.complete()

    metrics = []
    for i in range(0, env.num_rounds):
        metric = {
            "experiment_id": experiment_id,
            "batch_id": batch_id,
            "sim_id": internal_sim_id,
            "prompt_structure": prompt,
            "ps": setup.ps.name,
            "grid_size": grid_size,
            "num_rounds": setup.num_rounds,
            "round": i,
            "bias": env.bias[i],
            "bias_av": env.bias_av[i],
            "seed": seed
        }

        if setup:
            metric["game"] = setup.game
            metric = {**metric, **setup.llm.model_dump()}

        if setup.ps.params:
            ps_params = {f"ps__{key}": value for key, value in setup.ps.params}
            metric = {**metric, **ps_params}

        for ac in agent_config.configs:
            metric[f"opinion_group__{ac.config.params.get('bias_score')}"] =  ac.ratio

        metrics.append(metric)
    
    metric_df = pd.DataFrame.from_records(metrics)

    report_path = os.path.join(report_dir, f"{internal_sim_id}.csv")

    metric_df.to_csv(report_path)


def run_simulations(setup_path: str, run_one: bool = False, device: int = 0):
    setups = load_setups(setup_path)
    print(f"STARTING EXPERIMENT WITH {len(setups)} SETUPS")
    for setup in setups:
        used_llm = llm(setup.llm, device)
        threads = []
        experiment_id = str(uuid.uuid4())
        results = f"{setup.results_repo}/{setup.setup_name}--{experiment_id}"
        grid_options = setup.grid_size
        if grid_options and run_one:
            grid_options = [next(iter(grid_options), grid_options)]
        print(f"SETUP {setup.setup_name} HAS {len(grid_options)} GRID OPTIONS ({grid_options})")
        for size in grid_options:
            internal_batch_id = str(uuid.uuid4())
            combinations = list(itertools.product(setup.prompts, setup.bias_primer, setup.seeds))
            print(f"SETUP {setup.setup_name} | SIZE {size} HAS {len(combinations)} SIMULATION COMBINATIONS")
            if combinations and run_one:
                combinations = [combinations[0]]
            for prompt, bias, seed in combinations:
                t = threading.Thread(target=run_simulation, args=(
                    experiment_id,
                    used_llm,
                    internal_batch_id,
                    size,
                    prompt,
                    bias,
                    seed,
                    results,
                    setup
                ))
                threads.append(t)
                t.start()
                time.sleep(1)

        for t in threads:
            t.join()

    print("All simulations completed.")


if __name__ == "__main__":
    parser = parse_args()

    args = parser.parse_args()

    run_simulations(args.path, device=0)
