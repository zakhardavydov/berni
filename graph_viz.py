from berni.utils import init_llm_simulation, BerniPSConfig
from berni.ps import *


if __name__ == "__main__":

    ps = BerniPSConfig(name="random_grid")

    _, env = init_llm_simulation(None, agent_name="opinion_llm", prompt="./prompts/techmonks", strategy_name="riot", ps=ps, grid_size=5, controlled_agent=-1, controlled_target_bias=0.5, num_rounds=1000)

    print(ps._G)

    ps.viz("plots/base_gps.png")
