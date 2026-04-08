"""Dataset indexing utilities."""

from pathlib import Path


def list_images(root: str, suffix: str = ".png") -> list[Path]:
    """Recursively list image files."""
    return sorted(Path(root).rglob(f"*{suffix}"))
