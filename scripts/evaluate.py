import csv
import json
from pathlib import Path

from src.benchmarking.benchmark import Benchmark
from src.config import (
    BASE_MODEL,
    CPU_CORES,
    CPU_NAME,
    DATASET_NAME,
    DATASET_PATH,
    DO_SAMPLE,
    GPU,
    GRAPHS_DIR,
    LORA_PATH,
    MAX_NEW_TOKENS,
    METRICS_DIR,
    NUM_EVAL_SAMPLES,
    OPERATING_SYSTEM,
    RAM,
    REPORTS_DIR,
    TEMPERATURE,
)
from src.reporting.report_generator import ReportGenerator
from src.visualization.graph_generator import GraphGenerator


def main():

    # --------------------------------------------------
    # Run Benchmark
    # --------------------------------------------------

    benchmark = Benchmark(
        dataset_path=DATASET_PATH,
        base_model=BASE_MODEL,
        lora_path=LORA_PATH,
    )

    results, benchmark_example = benchmark.evaluate(
        NUM_EVAL_SAMPLES
    )

    # --------------------------------------------------
    # Print Results
    # --------------------------------------------------

    print("\nEvaluation Results")
    print("------------------")

    for metric, score in results.items():

        if isinstance(score, float):
            print(f"{metric}: {score:.4f}")
        else:
            print(f"{metric}: {score}")

    # --------------------------------------------------
    # Save JSON
    # --------------------------------------------------

    metrics_dir = Path(METRICS_DIR)
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

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    csv_file = metrics_dir / "evaluation_results.csv"

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(["Metric", "Value"])

        for metric, value in results.items():
            writer.writerow([metric, value])

    # --------------------------------------------------
    # Generate Markdown Report
    # --------------------------------------------------

    report_generator = ReportGenerator()

    report_generator.generate(
        metrics=results,
        model_name=BASE_MODEL,
        dataset_name=DATASET_NAME,
        num_samples=NUM_EVAL_SAMPLES,
        output_path=f"{REPORTS_DIR}/benchmark_report.md",

        hardware={
            "cpu": CPU_NAME,
            "cores": CPU_CORES,
            "ram": RAM,
            "gpu": GPU,
            "os": OPERATING_SYSTEM,
        },

        generation={
            "max_new_tokens": MAX_NEW_TOKENS,
            "sampling": DO_SAMPLE,
            "temperature": TEMPERATURE,
        },

        example=benchmark_example,
    )

    # --------------------------------------------------
    # Generate Graphs
    # --------------------------------------------------

    graph_generator = GraphGenerator()

    graphs_dir = Path(GRAPHS_DIR)
    graphs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph_generator.generate_quality_graph(
        metrics=results,
        output_path=str(
            graphs_dir / "quality_metrics.png"
        ),
    )

    graph_generator.generate_performance_graph(
        metrics=results,
        output_path=str(
            graphs_dir / "performance_metrics.png"
        ),
    )

    # --------------------------------------------------
    # Finished
    # --------------------------------------------------

    print(f"\nResults saved to: {metrics_file}")
    print(f"CSV results saved to: {csv_file}")
    print(
        f"Benchmark report saved to: "
        f"{REPORTS_DIR}/benchmark_report.md"
    )
    print(
        f"Quality graph saved to: "
        f"{graphs_dir / 'quality_metrics.png'}"
    )
    print(
        f"Performance graph saved to: "
        f"{graphs_dir / 'performance_metrics.png'}"
    )


if __name__ == "__main__":
    main()