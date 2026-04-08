"""Dataset tests."""

from mrti.data.dataset import list_images


def test_list_images_missing_dir_returns_empty() -> None:
    """Missing directory should return empty list."""
    assert list_images("does_not_exist") == []
