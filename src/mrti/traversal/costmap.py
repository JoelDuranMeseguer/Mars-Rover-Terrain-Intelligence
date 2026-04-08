"""Costmap creation placeholders."""


def build_costmap(mask: list[list[int]]) -> list[list[float]]:
    """Convert class ids to simple costs."""
    return [[float(v + 1) for v in row] for row in mask]
