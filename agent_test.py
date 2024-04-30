from berni.utils import init_llm_simulation, load_setups, parse_args, llm


if __name__ == "__main__":

    parser = parse_args()

    parser.add_argument("prompt", default="prompts/techmonks")

    args = parser.parse_args()

    conifgs = load_setups(args.path)
    assert conifgs, "No configs found"

    config = conifgs[0]
    
    used_llm = llm(config.llm, device=0)

    agents, env = init_llm_simulation(
        used_llm,
        "opinion_llm",
        args.prompt,
        config.grid_size[0],
        ps=config.ps,
        controlled_agent=-1,
        controlled_target_bias=0.2,
        num_rounds=10
    )

    env.setup()
    env.step()
