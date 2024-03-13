from typing import Any

from pydantic import BaseModel, Field

from langchain.chat_models import ChatOpenAI
from langchain.llms.base import BaseLanguageModel
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

from nypd import settings


n_gpu_layers = -1
n_batch = 512


class BerniLLMConfig(BaseModel):
    model_type: str
    model: str
    temp: float
    config: dict[str, Any] = {}


def llm(config: BerniLLMConfig) -> BaseLanguageModel | None:
    llm = None
    if config.model_type == "llama":
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = LlamaCpp(
            model_path=config.model,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            f16_kv=False,
            verbose=True,
            temperature=config.temp,
            callback_manager=callback_manager,
            max_tokens=24,
            n_threads=12
        )
        return llm
    elif config.model_type == "openai":
        llm = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY, model_name=config.model, temperature=config.temp, **config)
    return llm
