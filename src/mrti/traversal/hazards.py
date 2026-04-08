"""Hazard extraction placeholders."""


def detect_hazards(costmap: list[list[float]], threshold: float = 2.0) -> int:
    """Count cells above threshold."""
    return sum(1 for row in costmap for value in row if value > threshold)
