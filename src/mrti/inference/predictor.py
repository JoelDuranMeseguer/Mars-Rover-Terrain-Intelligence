"""Prediction placeholders."""


def predict(image_path: str) -> list[list[int]]:
    """Return a tiny dummy mask."""
    _ = image_path
    return [[0, 1], [1, 0]]
