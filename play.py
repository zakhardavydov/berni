import mlflow

from langchain.chat_models import ChatOpenAI

from nypd.core import settings
from nypd.driver import Driver
from nypd.structures import ExperimentSetup
from nypd.strategy import registry

from berni.agent import RiotLLMAgent
from berni.strategy import RiotPromptStrategy
from berni.swarm import PromptSwarmGenerator
from berni.ps import RandomGridPartnerSelection


AGENT_PROMPT_DIR = "./prompts/riot_reaper/agents"
STRATEGY_PROMPT_DIR = "./prompts/riot_reaper/strategy"
RULES_PROMPT_PATH = "./prompts/riot_reaper/rules.txt"

GRID_SIZE = 5
NUM_ROUNDS = 10

NORM = "reputation0"


if __name__ == "__main__":

    llm = ChatOpenAI(openai_api_key="sk-6s8nvmY2AvWkuMZ20zccT3BlbkFJhVrtHjV3dxn2PyoUrBTg", model_name="gpt-3.5-turbo-0613")

    mlflow.set_tracking_uri(settings.ML_FLOW_URL)
    mlflow.set_registry_uri(settings.ML_FLOW_URL)

    client = mlflow.client.MlflowClient(
        tracking_uri=settings.ML_FLOW_URL,
        registry_uri=settings.ML_FLOW_URL
    )

    generator = PromptSwarmGenerator(
        llm=llm,
        agent_prompt_dir=AGENT_PROMPT_DIR,
        strategy_prompt_dir=STRATEGY_PROMPT_DIR,
        rule_prompt_path=RULES_PROMPT_PATH,
        default_agent_type="opinion_llm"
    )

    ps = RandomGridPartnerSelection(grid_size=GRID_SIZE)

    agent_config, strategy = generator.generate()
    generator.register_strategy(registry, [(RiotLLMAgent, s) for s in strategy])

    exp_setup = ExperimentSetup(
        num_agents=ps.num_agents(),
        num_rounds=NUM_ROUNDS,
        game="pd",
        norm=NORM,
        config=agent_config
    )
    EXPERIMENT_NAME = f"zak/riot_reaper"

    Driver.play_with_setup(
        exp=exp_setup,
        exp_name=EXPERIMENT_NAME,
        seed=ps,
        ps=ps,
        ml_flow_client=client,
        initial_seed=42,
        number_of_times=1
    )
