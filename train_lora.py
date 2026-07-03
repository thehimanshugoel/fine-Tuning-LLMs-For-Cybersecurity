from datasets import load_from_disk
from transformers import Trainer, TrainingArguments

from src.training import (
    load_model_and_tokenizer,
    create_data_collator,
)

from src.peft_utils import (
    attach_lora,
    print_trainable_parameters,
)

DATASET_PATH = "data/processed/qwen_tokenized_dataset"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "outputs/lora"


def main():
    # Load processed dataset
    dataset = load_from_disk(DATASET_PATH)

    # Temporary: use only the first 100 examples
    train_dataset = dataset["train"].select(range(100))

    # Load base model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # Attach LoRA adapters
    model = attach_lora(model)

    # Print trainable parameter statistics
    print_trainable_parameters(model)

    # Create data collator
    data_collator = create_data_collator(tokenizer)

    # Training configuration
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
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

    # Save LoRA adapters
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    print("\nLoRA training completed successfully!")
    print(f"LoRA adapters saved to: {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()