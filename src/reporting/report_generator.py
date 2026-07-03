from datetime import datetime
from pathlib import Path


class ReportGenerator:
    """
    Generates benchmark reports.
    """

    def generate(
        self,
        metrics: dict,
        model_name: str,
        dataset_name: str,
        num_samples: int,
        output_path: str,
    ) -> None:

        report = f"""LLM Benchmark Report
====================

Generated
---------
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Model
-----
{model_name}

Dataset
-------
{dataset_name}

Samples Evaluated
-----------------
{num_samples}

Quality Metrics
---------------
Exact Match                : {metrics["exact_match"]:.4f}
BLEU                       : {metrics["bleu"]:.4f}
ROUGE-L                    : {metrics["rougeL"]:.4f}
BERTScore                  : {metrics["bertscore"]:.4f}

Performance Metrics
-------------------
Average Latency (seconds)  : {metrics["average_latency_seconds"]:.4f}
Average Generated Tokens   : {metrics["average_generated_tokens"]:.2f}
Tokens / Second            : {metrics["tokens_per_second"]:.4f}
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