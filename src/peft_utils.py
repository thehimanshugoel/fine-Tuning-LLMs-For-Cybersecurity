from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)


def create_lora_config():
    """
    Create and return the LoRA configuration.
    """

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )

    return config


def attach_lora(model):
    """
    Attach LoRA adapters to the model.
    """

    config = create_lora_config()

    model = get_peft_model(model, config)

    return model


def print_trainable_parameters(model):
    model.print_trainable_parameters()