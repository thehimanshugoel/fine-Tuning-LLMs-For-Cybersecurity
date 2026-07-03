"""
# Evaluation metrics for language model benchmarking.
"""


def compute_exact_match(prediction: str, reference: str) -> float:
    """
    Compute Exact Match (EM) between a prediction and reference.

    Returns:
        1.0 if the normalized strings match exactly, otherwise 0.0.
    """
    prediction = prediction.strip().lower()
    reference = reference.strip().lower()

    return float(prediction == reference)