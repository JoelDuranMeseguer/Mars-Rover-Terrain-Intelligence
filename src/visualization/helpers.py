"""Visualization helpers for terrain intelligence outputs."""

from typing import Iterable


def summarize_segmentation_map(segmentation_map: Iterable[Iterable[int]]) -> dict[int, int]:
    """Return class counts for a segmentation map.

    This avoids hard dependencies on plotting libraries in the initial scaffold.
    """
    counts: dict[int, int] = {}
    for row in segmentation_map:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    return counts
