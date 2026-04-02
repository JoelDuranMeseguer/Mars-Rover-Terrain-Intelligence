"""Segmentation training scaffold."""

from pathlib import Path


def train_segmentation_model(output_model_path: str, epochs: int = 5) -> Path:
    """Mock segmentation training routine.

    This starter writes a placeholder model file so downstream steps can run.
    """
    path = Path(output_model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"placeholder-segmentation-model\nepochs={epochs}\n", encoding="utf-8")
    return path
