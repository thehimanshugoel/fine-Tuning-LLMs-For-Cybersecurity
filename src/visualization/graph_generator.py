from pathlib import Path

import matplotlib.pyplot as plt


class GraphGenerator:
    """
    Generates benchmark graphs.
    """

    def generate_quality_graph(
        self,
        metrics: dict,
        output_path: str,
    ) -> None:

        quality_metrics = {
            "BLEU": metrics["bleu"],
            "ROUGE-L": metrics["rougeL"],
            "BERTScore": metrics["bertscore"],
        }

        plt.figure(figsize=(8, 5))

        bars = plt.bar(
            quality_metrics.keys(),
            quality_metrics.values(),
        )

        plt.title("Quality Metrics")
        plt.ylabel("Score")
        plt.ylim(0, 1)

        for bar in bars:
            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def generate_performance_graph(
        self,
        metrics: dict,
        output_path: str,
    ) -> None:

        performance_metrics = {
            "Latency (s)": metrics["average_latency_seconds"],
            "Tokens/sec": metrics["tokens_per_second"],
        }

        plt.figure(figsize=(8, 5))

        bars = plt.bar(
            performance_metrics.keys(),
            performance_metrics.values(),
        )

        plt.title("Performance Metrics")
        plt.ylabel("Value")

        for bar in bars:
            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()