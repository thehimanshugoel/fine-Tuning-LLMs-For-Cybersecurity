from src.inference import LoRAInference

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = "outputs/lora/final"


def main():
    inference = LoRAInference(
        base_model=BASE_MODEL,
        lora_path=LORA_PATH,
    )

    response = inference.generate(
        system="You are a cybersecurity expert.",
        user="Explain what SQL Injection is."
    )

    print(response)


if __name__ == "__main__":
    main()