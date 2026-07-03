"""
BLEU evaluation metric.
"""

import evaluate

bleu = evaluate.load("bleu")


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """
    Compute BLEU score.

    Args:
        predictions: Model predictions.
        references: Ground-truth responses.

    Returns:
        BLEU score.
    """

    results = bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references],
    )

    return results["bleu"]