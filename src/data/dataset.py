"""Dataset loading utilities."""

from pathlib import Path
from typing import List


def list_image_files(data_dir: str, suffixes: tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> List[Path]:
    """Return sorted image file paths from a directory.

    Args:
        data_dir: Directory containing image files.
        suffixes: Allowed file extensions.

    Returns:
        A sorted list of matching file paths.
    """
    base = Path(data_dir)
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.suffix.lower() in suffixes])
