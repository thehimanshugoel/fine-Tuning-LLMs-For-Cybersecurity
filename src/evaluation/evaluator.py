"""
Evaluation engine for language model benchmarking.
"""

from .metrics import compute_exact_match


class Evaluator:
    """Evaluates model predictions against reference answers."""

    def evaluate(self, predictions: list[str], references: list[str]) -> dict:
        """
        Evaluate multiple predictions.

        Args:
            predictions: List of model predictions.
            references: List of ground-truth answers.

        Returns:
            Dictionary containing aggregated evaluation metrics.
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        exact_match_scores = [
            compute_exact_match(pred, ref)
            for pred, ref in zip(predictions, references)
        ]

        return {
            "exact_match": sum(exact_match_scores) / len(exact_match_scores)
        }