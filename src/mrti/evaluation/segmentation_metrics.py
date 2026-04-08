"""Segmentation metrics placeholders."""


def pixel_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Compute 1D pixel accuracy."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)
