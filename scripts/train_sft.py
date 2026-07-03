from datasets import load_from_disk
from transformers import Trainer, TrainingArguments

from src.training import (
    load_model_and_tokenizer,
    create_data_collator,
)

DATASET_PATH = "data/processed/qwen_tokenized_dataset"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "outputs/sft"


def main():
    # Load processed dataset
    dataset = load_from_disk(DATASET_PATH)

    # Temporary: use only the first 100 examples
    train_dataset = dataset["train"]

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # Create data collator
    data_collator = create_data_collator(tokenizer)

    # Training configuration
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    print("\nTraining completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()