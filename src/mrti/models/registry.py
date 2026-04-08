"""Model registry."""


def available_models() -> list[str]:
    """Return supported model names."""
    return ["deeplab", "segformer"]
