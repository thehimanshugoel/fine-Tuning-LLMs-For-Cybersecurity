from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


def load_model_and_tokenizer(model_name: str):
    """
    Load the tokenizer and model for causal language modeling.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Qwen doesn't define a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer


def create_data_collator(tokenizer):
    """
    Create the data collator used during supervised fine-tuning.
    """

    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )