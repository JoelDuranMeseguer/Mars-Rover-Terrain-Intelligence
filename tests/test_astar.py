"""A* tests."""

from mrti.planning.astar import plan


def test_plan_returns_start_and_goal() -> None:
    """Placeholder planner should include endpoints."""
    assert plan((0, 0), (1, 1)) == [(0, 0), (1, 1)]
