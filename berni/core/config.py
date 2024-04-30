from pydantic_settings import BaseSettings
from dotenv import load_dotenv


class BerniSettings(BaseSettings):
    CURRENT_USER_DIR: str

    AGENT_PROMPT_DIR: str = "agents"
    STRATEGY_PROMPT_DIR: str = "strategy"
    RULES_PROMPT_PATH: str = "rules.txt"
    DEBATE_TOPIC_PATH: str = "debate.txt"
    PROMPT_DIR: str = "./prompts"
    DEFAULT_NORM: str = "reputation0"
    DEFAULT_MATRIX_GAME: str = "pd"


load_dotenv()
settings = BerniSettings()
