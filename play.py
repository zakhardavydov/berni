import mlflow

from langchain.chat_models import ChatOpenAI

from nypd.core import settings
from nypd.driver import Driver
from nypd.structures import ExperimentSetup
from nypd.strategy import registry

from berni.agent import LLMAgent
from berni.strategy import PromptStrategy
from berni.swarm import PromptSwarmGenerator


AGENT_PROMPT_DIR = "./prompts/pd/agents"
STRATEGY_PROMPT_DIR = "./prompts/pd/strategy"
RULES_PROMPT_PATH = "./prompts/pd/rules.txt"


NUM_AGENTS = 4
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
        rule_prompt_path=RULES_PROMPT_PATH
    )

    agent_config, strategy = generator.generate()
    generator.register_strategy(registry, [(LLMAgent, s) for s in strategy])

    print(agent_config)

    exp_setup = ExperimentSetup(
        num_agents=NUM_AGENTS,
        num_rounds=NUM_ROUNDS,
        game="pd",
        norm=NORM,
        config=agent_config
    )
    EXPERIMENT_NAME = f"zak/llm-test"

    Driver.play_with_setup(exp_setup, EXPERIMENT_NAME, client, initial_seed=42, number_of_times=1)
