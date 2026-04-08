"""Configuration utilities."""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load and parse a YAML configuration file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
