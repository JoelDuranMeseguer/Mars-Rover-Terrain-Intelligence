"""Planning metrics placeholders."""


def success_rate(successes: int, total: int) -> float:
    """Compute planning success rate."""
    return (successes / total) if total else 0.0
