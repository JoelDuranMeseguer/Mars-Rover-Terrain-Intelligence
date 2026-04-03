"""Inference entrypoint for terrain segmentation."""

import argparse

from src.models.inference import run_segmentation_inference
from src.navigation.traversability import compute_traversability_score


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run terrain segmentation inference.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--image", required=True, help="Path to image file.")
    return parser.parse_args()


def main() -> None:
    """Run inference and print simple output metrics."""
    args = parse_args()
    segmentation_map = run_segmentation_inference(args.image)
    score = compute_traversability_score(segmentation_map)
    height = len(segmentation_map)
    width = len(segmentation_map[0]) if height else 0

    print(f"Segmentation map shape: ({height}, {width})")
    print(f"Traversability score: {score:.3f}")


if __name__ == "__main__":
    main()
