"""Grid helpers."""


def shape(grid: list[list[float]]) -> tuple[int, int]:
    """Return (rows, cols) for grid."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    return rows, cols
