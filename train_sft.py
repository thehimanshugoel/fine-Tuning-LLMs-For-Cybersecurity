from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


DATASET_PATH = "data/processed/qwen_tokenized_dataset"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "outputs/sft"


def main():
    # Load processed dataset
    dataset = load_from_disk(DATASET_PATH)

    # Temporary: use only the first 100 examples
    dataset["train"] = dataset["train"].select(range(100))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

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
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save final model
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    print("\n Training completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()