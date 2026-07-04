import json
from pathlib import Path

from src.benchmarking.benchmark import Benchmark
from src.reporting.report_generator import ReportGenerator
from src.visualization.graph_generator import GraphGenerator


DATASET_PATH = "data/processed/qwen_tokenized_dataset"

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = "outputs/lora/final"

NUM_EVAL_SAMPLES = 10


def main():

    # Run benchmark
    benchmark = Benchmark(
        dataset_path=DATASET_PATH,
        base_model=BASE_MODEL,
        lora_path=LORA_PATH,
    )

    results = benchmark.evaluate(NUM_EVAL_SAMPLES)

    # Print Results
    print("\nEvaluation Results")
    print("------------------")

    for metric, score in results.items():

        if isinstance(score, float):
            print(f"{metric}: {score:.4f}")
        else:
            print(f"{metric}: {score}")

    # Save Metrics
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_file = metrics_dir / "evaluation_results.json"

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=4)

    # Generate Report
    report_generator = ReportGenerator()

    report_generator.generate(
        metrics=results,
        model_name="Qwen2.5-0.5B + LoRA",
        dataset_name="Trendyol Cybersecurity",
        num_samples=NUM_EVAL_SAMPLES,
        output_path="outputs/reports/benchmark_report.txt",
    )

    # Generate Graphs
    graph_generator = GraphGenerator()

    graph_generator.generate_quality_graph(
        metrics=results,
        output_path="outputs/graphs/quality_metrics.png",
    )

    graph_generator.generate_performance_graph(
        metrics=results,
        output_path="outputs/graphs/performance_metrics.png",
    )

    print(f"\nResults saved to: {metrics_file}")
    print("Benchmark report saved to: outputs/reports/benchmark_report.txt")
    print("Quality graph saved to: outputs/graphs/quality_metrics.png")
    print("Performance graph saved to: outputs/graphs/performance_metrics.png")


if __name__ == "__main__":
    main()