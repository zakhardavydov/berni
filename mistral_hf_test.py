import time
from transformers import AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


messages = [
    {"role": "user", "content": "Question: What NFL team won the Super Bowl in the year Justin Bieber was born? Answer: Let's work this out in a step by step way to be sure we have the right answer."}
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

start = time.time()
generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
tokenizer.batch_decode(generated_ids)[0]

end = time.time()

print(end - start)