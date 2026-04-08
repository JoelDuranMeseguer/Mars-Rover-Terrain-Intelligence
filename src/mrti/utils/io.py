"""I/O utility functions."""

from pathlib import Path


def ensure_dir(path: str) -> Path:
    """Create a directory if missing and return it."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
