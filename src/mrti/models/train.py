"""Training placeholders."""


def train(model_name: str, epochs: int) -> str:
    """Return a short training status string."""
    return f"trained {model_name} for {epochs} epochs"
