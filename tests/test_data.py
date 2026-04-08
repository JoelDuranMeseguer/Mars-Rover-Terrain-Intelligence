"""Placeholder tests for dataset utilities."""

from src.data.dataset import list_image_files


def test_list_image_files_missing_dir_returns_empty_list() -> None:
    """Dataset listing should handle missing folders gracefully."""
    assert list_image_files("does_not_exist") == []
