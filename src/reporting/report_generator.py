from datetime import datetime
from pathlib import Path


class ReportGenerator:
    """
    Generates benchmark reports in Markdown format.
    """

    def generate(
        self,
        metrics: dict,
        model_name: str,
        dataset_name: str,
        num_samples: int,
        output_path: str,
        hardware: dict,
        generation: dict,
        example: dict,
    ) -> None:

        report = f"""# LLM Benchmark Report

## Generated

{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

# Model

| Item | Value |
|------|-------|
| Base Model | {model_name} |
| Fine-tuning | LoRA |

---

# Dataset

| Item | Value |
|------|-------|
| Dataset | {dataset_name} |
| Evaluation Samples | {num_samples} |

---

# Hardware

| Component | Specification |
|----------|---------------|
| CPU | {hardware["cpu"]} |
| CPU Cores | {hardware["cores"]} |
| RAM | {hardware["ram"]} |
| GPU | {hardware["gpu"]} |
| Operating System | {hardware["os"]} |

---

# Generation Settings

| Setting | Value |
|---------|------:|
| Max New Tokens | {generation["max_new_tokens"]} |
| Sampling | {generation["sampling"]} |
| Temperature | {generation["temperature"]} |

---

# Quality Metrics

| Metric | Value |
|--------|------:|
| Exact Match | {metrics["exact_match"]:.4f} |
| BLEU | {metrics["bleu"]:.4f} |
| ROUGE-L | {metrics["rougeL"]:.4f} |
| BERTScore | {metrics["bertscore"]:.4f} |

---

# Performance Metrics

| Metric | Value |
|--------|------:|
| Average Latency (seconds) | {metrics["average_latency_seconds"]:.4f} |
| Average Generated Tokens | {metrics["average_generated_tokens"]:.2f} |
| Tokens / Second | {metrics["tokens_per_second"]:.4f} |

---

# Example Evaluation

## User Prompt

{example["prompt"]}

---

## Ground Truth

{example["reference"]}

---

## Model Prediction

{example["prediction"]}

---

# Generated Artifacts

- evaluation_results.json
- evaluation_results.csv
- benchmark_report.md
- quality_metrics.png
- performance_metrics.png
"""

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report)