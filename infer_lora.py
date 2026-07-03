import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = "outputs/lora/final"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    device_map="cpu",
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
)

model.eval()

prompt = "Explain what SQL Injection is."

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
    )

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
)

print(response)