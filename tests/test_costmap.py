"""Costmap tests."""

from mrti.traversal.costmap import build_costmap


def test_build_costmap_offsets_values() -> None:
    """Class ids should be shifted to positive costs."""
    assert build_costmap([[0, 1]]) == [[1.0, 2.0]]
