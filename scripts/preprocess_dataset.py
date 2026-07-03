from datasets import load_dataset
from transformers import AutoTokenizer

from src.preprocessing import (
    format_chat,
    is_valid,
    tokenize_function,
)


DATASET_NAME = "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "data/processed/qwen_tokenized_dataset"


def main():
    # Load dataset
    dataset = load_dataset(DATASET_NAME)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Validate dataset
    clean_dataset = dataset.filter(is_valid)

    # Create train/test split
    split_dataset = clean_dataset["train"].train_test_split(
        test_size=0.1,
        seed=42,
    )

    # Format chat conversations
    formatted_dataset = split_dataset.map(
        lambda example: format_chat(example, tokenizer)
    )

    # Tokenize dataset
    tokenized_dataset = formatted_dataset.map(
        lambda example: tokenize_function(example, tokenizer)
    )

    # Save processed dataset
    tokenized_dataset.save_to_disk(OUTPUT_DIR)

    print("Preprocessing completed successfully.")
    print(f"Dataset saved to: {OUTPUT_DIR}")

    print("\nDataset Structure")
    print(tokenized_dataset)

    print("\nTrain Columns")
    print(tokenized_dataset["train"].column_names)

    print("\nTest Columns")
    print(tokenized_dataset["test"].column_names)


if __name__ == "__main__":
    main()