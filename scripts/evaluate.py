import json
import time
from pathlib import Path

from datasets import load_from_disk

from src.evaluation.evaluator import Evaluator
from src.inference import LoRAInference
from src.reporting.report_generator import ReportGenerator


DATASET_PATH = "data/processed/qwen_tokenized_dataset"

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = "outputs/lora/final"

# Number of samples to evaluate
NUM_EVAL_SAMPLES = 10


def main():

    # Load test dataset
    dataset = load_from_disk(DATASET_PATH)

    test_dataset = dataset["test"].select(
        range(min(NUM_EVAL_SAMPLES, len(dataset["test"])))
    )

    # Load inference model
    inference = LoRAInference(
        base_model=BASE_MODEL,
        lora_path=LORA_PATH,
    )

    predictions = []
    references = []
    latencies = []
    token_counts = []

    print("Generating predictions...\n")

    for idx, example in enumerate(test_dataset):

        start_time = time.perf_counter()

        prediction = inference.generate(
            system=example["system"],
            user=example["user"],
        )

        end_time = time.perf_counter()

        latency = end_time - start_time
        latencies.append(latency)

        predictions.append(prediction)
        references.append(example["assistant"])

        generated_tokens = len(
            inference.tokenizer.encode(
                prediction,
                add_special_tokens=False,
            )
        )

        token_counts.append(generated_tokens)

        print(f"[{idx + 1}/{len(test_dataset)}] Done")

    # Performance metrics
    average_latency = sum(latencies) / len(latencies)
    average_tokens = sum(token_counts) / len(token_counts)
    tokens_per_second = average_tokens / average_latency

    # Quality metrics
    evaluator = Evaluator()

    results = evaluator.evaluate(
        predictions,
        references,
    )

    # Benchmark metrics
    results["average_latency_seconds"] = average_latency
    results["average_generated_tokens"] = average_tokens
    results["tokens_per_second"] = tokens_per_second

    print("\nEvaluation Results")
    print("------------------")

    for metric, score in results.items():
        if isinstance(score, float):
            print(f"{metric}: {score:.4f}")
        else:
            print(f"{metric}: {score}")

    # Save metrics JSON
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = metrics_dir / "evaluation_results.json"

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # Generate benchmark report
    report_generator = ReportGenerator()

    report_generator.generate(
        metrics=results,
        model_name="Qwen2.5-0.5B + LoRA",
        dataset_name="Trendyol Cybersecurity",
        num_samples=len(test_dataset),
        output_path="outputs/reports/benchmark_report.txt",
    )

    print(f"\nResults saved to: {metrics_file}")
    print("Benchmark report saved to: outputs/reports/benchmark_report.txt")


if __name__ == "__main__":
    main()