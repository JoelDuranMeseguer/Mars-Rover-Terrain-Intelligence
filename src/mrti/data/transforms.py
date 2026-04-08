"""Data transform placeholders."""


def normalize(values: list[float]) -> list[float]:
    """Normalize 0-255 values to 0-1."""
    if not values:
        return []
    return [v / 255.0 for v in values]
