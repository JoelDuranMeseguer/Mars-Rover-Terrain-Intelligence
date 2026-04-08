"""Metrics tests."""

from mrti.evaluation.segmentation_metrics import pixel_accuracy


def test_pixel_accuracy_basic() -> None:
    """Accuracy should be computed correctly."""
    assert pixel_accuracy([0, 1, 1], [0, 1, 0]) == 2 / 3
