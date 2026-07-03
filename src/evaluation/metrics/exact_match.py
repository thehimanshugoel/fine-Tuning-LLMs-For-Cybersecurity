"""
Exact Match evaluation metric.
"""


def compute_exact_match(prediction: str, reference: str) -> float:
    """
    Compute Exact Match (EM) between a prediction and reference.
    """
    prediction = prediction.strip().lower()
    reference = reference.strip().lower()

    return float(prediction == reference)