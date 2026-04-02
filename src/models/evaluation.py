"""Evaluation helpers for segmentation outputs."""

from typing import Iterable


def pixel_accuracy(y_true: Iterable[Iterable[int]], y_pred: Iterable[Iterable[int]]) -> float:
    """Compute pixel-wise accuracy."""
    true_rows = [list(r) for r in y_true]
    pred_rows = [list(r) for r in y_pred]
    if len(true_rows) != len(pred_rows) or not true_rows:
        return 0.0

    total = 0
    correct = 0
    for t_row, p_row in zip(true_rows, pred_rows):
        if len(t_row) != len(p_row):
            return 0.0
        for t_val, p_val in zip(t_row, p_row):
            total += 1
            if t_val == p_val:
                correct += 1

    return (correct / total) if total else 0.0
