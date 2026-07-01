from datasets import load_dataset
from transformers import AutoTokenizer

from src.preprocessing import (
    format_chat,
    is_valid,
    tokenize_function,
)


dataset = load_dataset(
    "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"
)

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)


clean_dataset = dataset.filter(is_valid)

formatted_dataset = clean_dataset.map(
    lambda example: format_chat(example, tokenizer)
)

tokenized_dataset = formatted_dataset.map(
    lambda example: tokenize_function(example, tokenizer)
)


print(tokenized_dataset)
print(tokenized_dataset["train"].column_names)
print(tokenized_dataset["train"][0]["input_ids"][:20])
print(tokenized_dataset["train"][0]["attention_mask"][:20])