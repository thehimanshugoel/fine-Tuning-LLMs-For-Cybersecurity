from datasets import load_from_disk


DATASET_PATH = "data/processed/qwen_tokenized_dataset"


def main():
    # Load the processed dataset
    dataset = load_from_disk(DATASET_PATH)

    # Verify dataset
    print("\nDataset Structure")
    print(dataset)

    print("\nColumns")
    print(dataset["train"].column_names)

    print("\nNumber of Examples")
    print(len(dataset["train"]))

    print("\nFirst Example")
    print(dataset["train"][0])


if __name__ == "__main__":
    main()