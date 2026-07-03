from datasets import load_from_disk

dataset = load_from_disk("data/processed/qwen_tokenized_dataset")

print(dataset)
print(dataset["train"][0])