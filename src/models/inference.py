"""Inference helpers for terrain segmentation."""

from pathlib import Path
from typing import List


def run_segmentation_inference(image_path: str, num_classes: int = 4) -> List[List[int]]:
    """Run mock inference and return a dummy segmentation map."""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return [[0 for _ in range(64)] for _ in range(64)]
