from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("\nModel Loaded")
print(model.__class__.__name__)

print("\nTokenizer Loaded")
print(tokenizer.__class__.__name__)