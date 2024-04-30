from typing import Any, Dict, Iterator, List, Mapping, Optional

from pydantic import BaseModel, Field

import torch

from langchain.chat_models import ChatOpenAI
from langchain.llms.base import BaseLanguageModel
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig

from nypd import settings


n_gpu_layers = -1
n_batch = 512


class BerniLLMConfig(BaseModel):
    model_type: str
    model: str
    temp: float
    max_tokens: int = 32
    batch_size: int = 64
    config: dict[str, Any] = {}


def llm(config: BerniLLMConfig, device: int) -> BaseLanguageModel | None:
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
            max_tokens=config.max_tokens,
            n_threads=12
        )

    elif config.model_type == "hf":
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(config.model)
        model = AutoModelForCausalLM.from_pretrained(config.model, quantization_config=nf4_config, device_map="auto")
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=config.max_tokens
        )
        llm = HuggingFacePipeline(pipeline=pipe, batch_size=config.batch_size)

    elif config.model_type == "openai":
        llm = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY, model_name=config.model, temperature=config.temp, **config)
        
    return llm
