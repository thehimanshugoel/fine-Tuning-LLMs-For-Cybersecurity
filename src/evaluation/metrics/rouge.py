"""
ROUGE evaluation metric.
"""

import evaluate

rouge = evaluate.load("rouge")


def compute_rouge(predictions: list[str], references: list[str]) -> float:
    """
    Compute ROUGE-L score.
    """

    results = rouge.compute(
        predictions=predictions,
        references=references,
    )

    return results["rougeL"]