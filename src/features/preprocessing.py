"""Preprocessing helpers for rover terrain images."""

from typing import Iterable, List


def normalize_image(image: Iterable[Iterable[float]]) -> List[List[float]]:
    """Normalize nested pixel values to range [0, 1]."""
    rows = [list(row) for row in image]
    flat = [value for row in rows for value in row]
    if not flat:
        return rows
    max_val = max(flat)
    divisor = 255.0 if max_val > 1.0 else 1.0
    return [[float(v) / divisor for v in row] for row in rows]
