from berni.utils import init_llm_simulation, BerniPSConfig
from berni.ps import *


if __name__ == "__main__":

    seeds = [42, 69, 1337]
    grid_sizes = [5, 8, 10]

    topology_params = [
        {
            "name": "erdos_renyi",
            "params": {
                "p": 0.1
            }
        },
        {
            "name": "erdos_renyi",
            "params": {
                "p": 0.2
            }
        },
        {
            "name": "erdos_renyi",
            "params": {
                "p": 0.3
            }
        },
        {
            "name": "watts_stogartz",
            "params": {
                "k": 3,
                "beta": 0.1
            }
        },
        {
            "name": "watts_stogartz",
            "params": {
                "k": 3,
                "beta": 0.2
            }
        },
        {
            "name": "watts_stogartz",
            "params": {
                "k": 4,
                "beta": 0.1
            }
        },
        {
            "name": "watts_stogartz",
            "params": {
                "k": 4,
                "beta": 0.2
            }
        },
        {
            "name": "barbasi_albert",
            "params": {
                "m": 3
            }
        },
        {
            "name": "barbasi_albert",
            "params": {
                "m": 7
            }
        },
        {
            "name": "barbasi_albert",
            "params": {
                "m": 10
            }
        }
    ]

    topology_params = [
        {
            "name": "barbasi_albert",
            "params": {
                "m": 12
            }
        }
    ]

    for grid_size in grid_sizes:
        for s in seeds:
            for t in topology_params:
                ps = BerniPSConfig(name=t.get("name"), params=t.get("params"))

                _, _, topology = init_llm_simulation(
                    None,
                    agent_name="opinion_llm",
                    prompt="./prompts/techmonks",
                    strategy_name="riot",
                    ps=ps,
                    grid_size=grid_size,
                    controlled_agent=-1,
                    controlled_target_bias=0.5,
                    num_rounds=1000,
                    seed=s
                )

                param_str = ""
                for param_key, param_value in ps.params.items():
                    param_str += f"{param_key}={param_value}__"

                format_params = ps.params

                topology.viz(f"plots/topology/{ps.name}__{param_str}{s}__{grid_size}.png")
