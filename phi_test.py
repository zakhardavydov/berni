from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from pynvml import *
from datasets import load_dataset
import time, torch

from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead


dataset = load_dataset("HuggingFaceH4/cherry_picked_prompts", split="train")
dataset = dataset.rename_column("prompt", "query")
dataset = dataset.remove_columns(["meta", "completion"])


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


base_model_id = "microsoft/phi-1_5"


compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_id, trust_remote_code=True, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

print(print_gpu_utilization())

config = PPOConfig(
    model_name=base_model_id,
    learning_rate=1.41e-5,
)

tokenizer.pad_token = tokenizer.eos_token

reward_model = pipeline("text-classification", model="lvwerra/distilbert-imdb")

def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample

dataset = dataset.map(tokenize, batched=False)
print(len(dataset))

ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

from tqdm import tqdm


epochs = 10
for epoch in tqdm(range(epochs), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader): 
        print(batch)
        query_tensors = batch["input_ids"]
    
        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    
        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    
        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

#### Save model
ppo_trainer.save_model("my_ppo_model")