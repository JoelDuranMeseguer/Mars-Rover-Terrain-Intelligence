"""Traversability scoring logic."""

from typing import Iterable


def compute_traversability_score(segmentation_map: Iterable[Iterable[int]]) -> float:
    """Compute a simple traversability score from class IDs."""
    values = [v for row in segmentation_map for v in row]
    if not values:
        return 0.0
    avg = sum(values) / len(values)
    max_class = max(values)
    denom = (max_class + 1.0) if max_class > 0 else 1.0
    score = 1.0 - (avg / denom)
    return max(0.0, min(1.0, score))
