"""
Evaluation engine for language model benchmarking.
"""

from .metrics.exact_match import compute_exact_match
from .metrics.bleu import compute_bleu
from .metrics.rouge import compute_rouge
from .metrics.bertscore import compute_bertscore

class Evaluator:
    """Evaluates model predictions against reference answers."""

    def evaluate(self, predictions: list[str], references: list[str]) -> dict:

        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        exact_match_scores = [
            compute_exact_match(pred, ref)
            for pred, ref in zip(predictions, references)
        ]

        return {
            "exact_match": sum(exact_match_scores) / len(exact_match_scores),
            "bleu": compute_bleu(predictions, references),
            "rougeL": compute_rouge(predictions, references),
            "bertscore": compute_bertscore(predictions, references),

        }