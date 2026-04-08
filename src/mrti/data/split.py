"""Dataset splitting placeholders."""


def split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Return train, val, test counts."""
    train = int(total * ratios[0])
    val = int(total * ratios[1])
    test = total - train - val
    return train, val, test
