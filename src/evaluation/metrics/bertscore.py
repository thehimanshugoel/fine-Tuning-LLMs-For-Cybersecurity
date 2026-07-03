"""
BERTScore evaluation metric.
"""

from bert_score import score


def compute_bertscore(
    predictions: list[str],
    references: list[str],
) -> float:
    """
    Compute average BERTScore (F1).
    """

    _, _, f1 = score(
        predictions,
        references,
        lang="en",
        verbose=False,
    )

    return f1.mean().item()