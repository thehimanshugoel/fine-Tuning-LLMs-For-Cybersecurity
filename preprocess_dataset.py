from datasets import load_dataset
from transformers import AutoTokenizer

# -----------------------------
# Load Dataset
# -----------------------------
dataset = load_dataset(
    "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"
)

# -----------------------------
# Load Tokenizer
# -----------------------------
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# -----------------------------
# Validate Dataset
# -----------------------------
def is_valid(example):
    return (
        example["system"] is not None
        and example["user"] is not None
        and example["assistant"] is not None
    )


print("Original dataset size:", len(dataset["train"]))

clean_dataset = dataset.filter(is_valid)

print("Clean dataset size:", len(clean_dataset["train"]))


# -----------------------------
# Format Chat
# -----------------------------
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


# -----------------------------
# Apply Preprocessing
# -----------------------------
formatted_dataset = clean_dataset.map(format_chat)


# -----------------------------
# Verify Output
# -----------------------------
print("\nDataset Structure\n")
print(formatted_dataset)

print("\nColumns\n")
print(formatted_dataset["train"].column_names)

print("\nFormatted Example\n")
print(formatted_dataset["train"][0]["text"][:1000])