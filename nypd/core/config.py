from pydantic_settings import BaseSettings
from dotenv import load_dotenv


class Settings(BaseSettings):
    ML_FLOW_URL: str
    CURRENT_USER_DIR: str
    OPENAI_API_KEY: str | None = None


load_dotenv()

settings = Settings()
