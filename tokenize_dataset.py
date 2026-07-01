from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset(
    "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"
)

model_name = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def is_valid(example):
    return (
        example["system"] is not None
        and example["user"] is not None
        and example["assistant"] is not None
    )

clean_dataset = dataset.filter(is_valid)

def format_chat(example):
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


formatted_dataset = clean_dataset.map(format_chat)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
    )

tokenized_dataset = formatted_dataset.map(tokenize_function)

print(tokenized_dataset)

print(tokenized_dataset["train"].column_names)

print(tokenized_dataset["train"][0]["input_ids"][:20])

print(tokenized_dataset["train"][0]["attention_mask"][:20])