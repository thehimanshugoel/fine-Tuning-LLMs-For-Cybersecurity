from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Explain SQL Injection."

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(**inputs, max_new_tokens=100)

response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)