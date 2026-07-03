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

        plt.bar(
            quality_metrics.keys(),
            quality_metrics.values(),
        )

        plt.title("Quality Metrics")
        plt.ylabel("Score")
        plt.ylim(0, 1)

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()