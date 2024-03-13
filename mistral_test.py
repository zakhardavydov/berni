from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

from berni.utils import llm, BerniLLMConfig


model = "/Users/zakdavydov/.cache/lm-studio/models/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate.from_template(template)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = -1
n_batch = 512

config = BerniLLMConfig(model_type="llama", model=model, temp=0)
used_llm = llm(config)

llm_chain = LLMChain(prompt=prompt, llm=used_llm)
question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
output = llm_chain.run(question)
print(output)
