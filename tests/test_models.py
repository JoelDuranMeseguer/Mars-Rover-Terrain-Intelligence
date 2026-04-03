"""Placeholder tests for model utilities."""

from src.models.evaluation import pixel_accuracy


def test_pixel_accuracy_perfect_prediction() -> None:
    """Perfect predictions should produce accuracy of 1.0."""
    y_true = [[0, 1], [1, 0]]
    y_pred = [[0, 1], [1, 0]]
    assert pixel_accuracy(y_true, y_pred) == 1.0
