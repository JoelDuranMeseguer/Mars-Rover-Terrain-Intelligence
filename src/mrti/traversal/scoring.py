"""Traversability scoring placeholders."""


def traversability_score(costmap: list[list[float]]) -> float:
    """Compute inverse mean cost score in [0,1]."""
    values = [v for row in costmap for v in row]
    if not values:
        return 0.0
    mean_cost = sum(values) / len(values)
    return max(0.0, min(1.0, 1.0 / mean_cost))
