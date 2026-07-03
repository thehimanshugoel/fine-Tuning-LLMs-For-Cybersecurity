import json
from pathlib import Path

from datasets import load_from_disk

from src.evaluation.evaluator import Evaluator
from src.inference import LoRAInference


DATASET_PATH = "data/processed/qwen_tokenized_dataset"

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = "outputs/lora/final"


def main():

    # Load test dataset
    dataset = load_from_disk(DATASET_PATH)

    # Temporary: evaluate only first 10 samples
    test_dataset = dataset["test"].select(range(10))

    # Load inference model
    inference = LoRAInference(
        base_model=BASE_MODEL,
        lora_path=LORA_PATH,
    )

    predictions = []
    references = []

    print("Generating predictions...\n")

    for idx, example in enumerate(test_dataset):

        prediction = inference.generate(
            system=example["system"],
            user=example["user"],
        )

        predictions.append(prediction)
        references.append(example["assistant"])

        print(f"[{idx + 1}/{len(test_dataset)}] Done")

    # Evaluate
    evaluator = Evaluator()

    results = evaluator.evaluate(
        predictions,
        references,
    )

    print("\nEvaluation Results")
    print("------------------")

    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

    # Save metrics
    output_dir = Path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "evaluation_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()