def is_valid(example):
    """
    Check whether a dataset example contains all required fields.
    """
    return (
        example["system"] is not None
        and example["user"] is not None
        and example["assistant"] is not None
    )


def format_chat(example, tokenizer):
    """
    Format a dataset example using the model's chat template.
    """
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


def tokenize_function(example, tokenizer):
    """
    Tokenize the formatted chat text.
    """
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
    )