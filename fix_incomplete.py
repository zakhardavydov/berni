import os
import math
import json
import argparse
import pandas as pd

from typing import Any


def crirsis_hero_prompt_decider(round: dict[str, Any]) -> str | None:
    prompts = []
    for agent_id, doc in round.items():
        prompts.append(doc["outcome_opinion"])
    all_prompts = ".".join(prompts)

    if "flint" in all_prompts.lower():
        return "prompts/crisishero_flint"
    return "promtps/crisishero_katrina"


def monks_prompt_decider(round: dict[str, Any]) -> str | None:
    prompts = []
    for agent_id, doc in round.items():
        if doc["round_action"] is not None:
            continue
        prompts.append(doc["round_prompt"])
    prompt_len = [len(prompt) for prompt in prompts]
    av = sum(prompt_len) / len(prompt_len)
    if av > 600:
        return "prompts/techmonks"
    return "prompts/techmonks_1"


def extract_num_agents(data):
    return len(data.keys())


def decide_ps(base_path):
    segments = base_path.split('_')
    
    target_segments = {'ey', 'ws', 'ba'}
    
    for segment in segments:
        if segment in target_segments:
            return segment
    
    return 'random_grid'


def decide_model(base_path) -> tuple[str, str]:
    if "gpt3.5" in base_path:
        return "openai", "gpt3.5"
    elif "mistral" in base_path:
        return "llama", "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    elif "llama3":
        return "llama", "./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    else:
        raise ValueError("Unknown model")


def count_party(data):
    bias_scores = [entry["initial_bias"] for entry in data.values()]
    unique_bias_scores = set(bias_scores)
    bias_score_count = {score: bias_scores.count(score) for score in unique_bias_scores}
    total_entries = len(bias_scores)
    bias_score_ratio = {score: count / total_entries for score, count in bias_score_count.items()}

    return {f"opinion_group__{bias_score}": round(ratio, 1) for bias_score, ratio in bias_score_ratio.items()}


def fix_incomplete(base_path: str):
    experiment_id = base_path.split("--")[1]
    ps = decide_ps(base_path)
    model_type, model = decide_model(base_path)
                
    for batch_id in os.listdir(base_path):
        batch_path = os.path.join(base_path, batch_id)
        if not os.path.isdir(batch_path):
            continue
        if not os.path.isdir(os.path.join(base_path, batch_id)):
            continue

        for simulation_id in os.listdir(batch_path):
            simulation_path = os.path.join(batch_path, simulation_id)
            
            if not os.path.isdir(simulation_path):
                continue
            
            matrix_path = os.path.join(simulation_path, "matrix")
            
            round_files = [f for f in os.listdir(matrix_path) if f.endswith('.json')]
            if not round_files:
                continue

            first_round_path = os.path.join(matrix_path, sorted(round_files)[0])

            with open(first_round_path, 'r') as file:
                data = json.load(file)
            
            num_agents = extract_num_agents(data)

            if "crisishero" in base_path:
                prompt_structure = crirsis_hero_prompt_decider(data)
            else:
                prompt_structure = monks_prompt_decider(data)

            parties = count_party(data)

            base_row = {
                'experiment_id': experiment_id,
                'batch_id': batch_id,
                'sim_id': simulation_id,
                'grid_size': int(math.sqrt(num_agents)),
                'prompt_structure': prompt_structure,
                'ps': ps,
                'model_type': model_type,
                'model': model
            }

            base_row = {**base_row, **parties}

            df = pd.DataFrame([base_row])
            csv_path = os.path.join(batch_path, f"{simulation_id}.csv")
            df.to_csv(csv_path, index=False)
            print(f"CSV created at {csv_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Fix incomplete experiment that finished early',
        description='Fix incomplete'
    )
    parser.add_argument("path")
    
    args = parser.parse_args()
    
    fix_incomplete(base_path=args.path)
